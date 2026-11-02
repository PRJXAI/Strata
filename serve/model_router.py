"""One API for several run configs, with at most one resident model."""
from __future__ import annotations

import argparse
import collections
import contextlib
import http.client
import json
import os
import secrets
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlsplit

from serve import lifeline
from serve.frontend import ChatTemplate
from serve.server import (ROOT, BadBody, ByteTokenizer, MockEngine, Server, Service,
                          V1_V1_ALIASES,
                          allowed_hosts_of, api_key_of, explicit_no_key, host_names_for,
                          make_handler, origins_of)

GENERATION_PATHS = {"/v1/chat/completions", "/v1/responses", "/v1/messages",
                    "/v1/messages/count_tokens"}
CONTROL_PATHS = {"/v1/load", "/v1/unload"}


class RouterError(Exception):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


def read_models(path):
    path = Path(path).resolve()
    cfg = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(cfg, dict) or not isinstance(cfg.get("models"), dict) or not cfg["models"]:
        raise ValueError('router config needs a nonempty "models": {"model-id": "run-config.json"}')
    models = {}
    for name, source in cfg["models"].items():
        if not isinstance(name, str) or not name.strip() or len(name) > 200 or any(ord(c) < 32 for c in name):
            raise ValueError("model IDs must be nonempty strings of at most 200 characters without control characters")
        if not isinstance(source, str) or not source:
            raise ValueError(f"model {name}: expected a run-config path")
        source = (path.parent / source).resolve()
        model = json.loads(source.read_text(encoding="utf-8-sig"))
        if not isinstance(model, dict) or "models" in model:
            raise ValueError(f"model {name}: expected an ordinary Strata run config")
        models[name] = (source, model)
    return cfg, models


class ModelRouter:
    def __init__(self, models, *, engine="strata", max_queue=32):
        if type(max_queue) is not int or not 0 <= max_queue <= 4096:
            raise ValueError("max_queue must be a whole number in 0..4096")
        self.models, self.engine, self.max_queue = models, engine, max_queue
        self.names = {name: name for name in models}
        self.aliases = {}
        for name, (_, model) in models.items():
            meta = SimpleNamespace(model=name)
            Service.set_aliases(meta, model.get("aliases"))
            original = model.get("model_name")
            if isinstance(original, str) and original.strip() and original != name and original not in meta.aliases:
                meta.aliases.append(original)
            for alias in meta.aliases:
                if alias in self.names and self.names[alias] != name:
                    raise ValueError(f"model name or alias {alias!r} belongs to more than one model")
                self.names[alias] = name
            self.aliases[name] = meta.aliases
        self.condition = threading.Condition()
        self.waiters = collections.deque()
        self.busy = self.closing = False
        self.active = None
        self.proc = self.port = None
        self.key = secrets.token_hex(32)
        self.temp = tempfile.TemporaryDirectory(prefix="strata-model-router-")
        self.directory = Path(self.temp.name)
        self.last_load_s = None

    @contextlib.contextmanager
    def turn(self, cancel):
        ticket = object()
        with self.condition:
            if self.closing:
                raise RouterError(503, "router is stopping")
            if len(self.waiters) + int(self.busy) >= self.max_queue + 1:
                raise RouterError(429, "model router queue is full; retry later")
            self.waiters.append(ticket)
            try:
                while self.busy or self.waiters[0] is not ticket:
                    if self.closing:
                        raise RouterError(503, "router is stopping")
                    if cancel.is_set():
                        raise RouterError(499, "client disconnected while queued")
                    self.condition.wait(0.1)
                if self.closing:
                    raise RouterError(503, "router is stopping")
                if cancel.is_set():
                    raise RouterError(499, "client disconnected while queued")
                self.busy = True
            finally:
                self.waiters.remove(ticket)
                self.condition.notify_all()
        try:
            yield
        finally:
            with self.condition:
                self.busy = False
                self.condition.notify_all()

    def connection(self, timeout=600):
        if self.port is None:
            raise RouterError(503, "no model server is running")
        return http.client.HTTPConnection("127.0.0.1", self.port, timeout=timeout)

    def control(self, path):
        conn = self.connection()
        try:
            conn.request("POST", path, b"{}", {"Content-Type": "application/json", "Authorization": f"Bearer {self.key}"})
            response = conn.getresponse()
            data = json.loads(response.read())
            if response.status != 200:
                raise RouterError(503, f"model control failed: {data.get('error', {}).get('message', response.status)}")
            return data
        finally:
            conn.close()

    def stop_model(self):
        proc = self.proc
        if proc is None:
            return
        # A disconnected stream may still be cancelling inside the backend. Never load its replacement first.
        if proc.poll() is None:
            try:
                if self.port is not None:
                    self.control("/unload?wait=60")
            except (OSError, ValueError, http.client.HTTPException, RouterError):
                pass
            proc.terminate()
            try:
                proc.wait(timeout=45)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)
        lifeline.release(proc)
        self.proc = self.port = self.active = None

    def load(self, name, cancel):
        if self.active == name and self.proc is not None and self.proc.poll() is None:
            self.control("/load")
            return
        self.stop_model()
        if cancel.is_set():
            raise RouterError(499, "client disconnected before loading")
        started = time.monotonic()
        source, model = self.models[name]
        model = {**model, "model_name": name, "aliases": self.aliases[name], "open_browser": False, "idle_unload_s": 0}
        config = self.directory / "model.json"
        config.write_text(json.dumps(model), encoding="utf-8")
        shared = source.with_name(source.stem + ".shared-settings.json")
        target = self.directory / "model.shared-settings.json"
        if shared.exists():
            target.write_bytes(shared.read_bytes())
        else:
            target.unlink(missing_ok=True)
        record = self.directory / "ready.json"
        record.unlink(missing_ok=True)
        env = {**os.environ, "STRATA_API_KEY": self.key}
        env.pop("STRATA_TELEMETRY_KEY", None)
        env.pop("STRATA_OWNER_ATTEST_KEY", None)
        # The backend stays on a private loopback port with an unguessable credential.
        args = [sys.executable, str(ROOT / "serve/server.py"), "--engine", self.engine,
                "--config", str(config), "--host", "127.0.0.1", "--port", "0", "--lazy",
                "--ready-file", str(record), "--shutdown-timeout", "30"]
        self.proc = lifeline.popen(args, role="engine", cwd=str(ROOT), env=env)
        try:
            deadline = time.monotonic() + 120
            data = None
            while data is None:
                if cancel.is_set():
                    raise RouterError(499, "client disconnected while starting model")
                if self.proc.poll() is not None:
                    raise RouterError(503, f"model {name} server exited with code {self.proc.returncode}; see server log")
                if time.monotonic() >= deadline:
                    raise RouterError(503, f"model {name} server did not become ready")
                try:
                    data = json.loads(record.read_text())
                except (FileNotFoundError, ValueError):
                    pass
                time.sleep(0.05)
            if data["pid"] != self.proc.pid or data["model_id"] != name:
                raise RouterError(503, "backend ready record does not match its process and model")
            self.port = data["bound_port"]
            self.control("/v1/load")
            self.active = name
            self.last_load_s = round(time.monotonic() - started, 3)
            print(f"[strata router] loaded {name} in {self.last_load_s:.3f} s", flush=True)
        except BaseException:
            self.stop_model()
            raise

    def status(self):
        proc = self.proc
        port = self.port
        live = proc is not None and proc.poll() is None
        active = self.active if live and port is not None else None
        if active is not None:
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=1)
            try:
                conn.request("GET", "/health", headers={"Authorization": f"Bearer {self.key}"})
                response = conn.getresponse()
                if response.status != 200 or not json.loads(response.read()).get("loaded"):
                    active = None
            except (OSError, ValueError, http.client.HTTPException):
                active = None
            finally:
                conn.close()
            if proc is not self.proc:
                active = None
        return {"mode": "one_resident_model", "active_model": active,
                "busy": self.busy, "queued": len(self.waiters), "last_load_s": self.last_load_s}

    def close(self):
        with self.condition:
            self.closing = True
            self.condition.notify_all()
            # Loading and non-streamed inference may be blocked on backend I/O. Parent-death protection also ends its engine.
            if self.busy and self.proc is not None and self.proc.poll() is None:
                self.proc.kill()
            while self.busy:
                self.condition.wait(0.1)
        self.stop_model()
        self.temp.cleanup()


def make_router_handler(router, guard):
    class Handler(make_handler(guard)):
        def do_GET(self):
            if not self._authorized():
                return
            path = urlsplit(self.path).path.rstrip("/")
            if path in ("/v1/models", "/models"):
                active = router.status()["active_model"]
                self._json(200, {"object": "list", "data": [
                    {"id": name, "object": "model", "aliases": router.aliases[name],
                     "status": {"value": "loaded" if name == active else "unloaded"}}
                    for name in router.models]})
            elif path in ("/health", "/api/health", "/v1/status"):
                self._json(200, {"status": "ok", **router.status()})
            else:
                self._json(404, {"error": {"message": "not found"}})

        def do_POST(self):
            if not self._authorized() or self._foreign_page():
                return
            path = urlsplit(self.path).path.rstrip("/")
            path = V1_V1_ALIASES.get(path, path)
            if path not in GENERATION_PATHS | CONTROL_PATHS:
                self._json(404, {"error": {"message": "not found"}})
                return
            if path in CONTROL_PATHS and not self._own_page("models can be loaded or unloaded"):
                return
            cancel = threading.Event()
            try:
                body = self._body()
                req = json.loads(body or b"{}")
                if not isinstance(req, dict):
                    raise RouterError(400, "send a JSON object")
                requested = req.get("model")
                name = router.names.get(requested) if isinstance(requested, str) else None
                if name is None:
                    raise RouterError(404, "model not found; select an ID from /v1/models")
                self._watch_client(cancel)
                with router.turn(cancel):
                    if path == "/v1/unload":
                        if router.active == name:
                            router.stop_model()
                        self._json(200, {"status": "unloaded", "model": name})
                    else:
                        router.load(name, cancel)
                        if path == "/v1/load":
                            self._json(200, {"status": "loaded", "model": name})
                        elif not cancel.is_set():
                            self._forward(router, path, body, cancel)
            except BadBody as e:
                self._error(e.status, str(e))
            except (ValueError, TypeError) as e:
                self._error(400, str(e))
            except RouterError as e:
                if e.status != 499:
                    self._error(e.status, str(e))
            except (OSError, http.client.HTTPException) as e:
                if not cancel.is_set():
                    self._error(502, f"model connection failed: {type(e).__name__}")
            finally:
                if self.watch_done is not None:
                    self.watch_done.set()
                with guard.status_lock:
                    guard.active_cancels.discard(cancel)

        def _error(self, status, message):
            if not self.answer_started:
                self._json(status, {"error": {"type": "server_error" if status >= 500 else "invalid_request_error",
                                              "message": message}}, retry_after=1 if status in (429, 503) else None)

        def _forward(self, router, path, body, cancel):
            conn = router.connection(timeout=3600)
            done = threading.Event()
            headers = {"Authorization": f"Bearer {router.key}", "Content-Type": "application/json"}
            for name in ("anthropic-version", "anthropic-beta"):
                if self.headers.get(name):
                    headers[name] = self.headers[name]
            if self._foreign_origin():
                # Keep all backend browser restrictions, including Responses images and MCP tools.
                headers["Origin"] = "https://router-untrusted.invalid"
            try:
                conn.connect()
                sock = conn.sock

                def abort_on_disconnect():
                    while not done.wait(0.1):
                        if cancel.is_set():
                            try:
                                sock.shutdown(socket.SHUT_RDWR)
                            except OSError:
                                pass
                            return

                threading.Thread(target=abort_on_disconnect, daemon=True).start()
                conn.request("POST", path, body, headers)
                response = conn.getresponse()
                self.send_response(response.status)
                for key, value in response.getheaders():
                    if key.lower() in ("content-type", "content-length", "retry-after", "cache-control", "x-accel-buffering"):
                        self.send_header(key, value)
                self._cors()
                self.end_headers()
                while not cancel.is_set():
                    block = response.read1(65536)
                    if not block:
                        break
                    self.wfile.write(block)
                    self.wfile.flush()
            finally:
                done.set()
                conn.close()

    return Handler


def start_router(cfg, models, *, host="127.0.0.1", port=8095, engine="strata"):
    tok = ByteTokenizer()
    guard = Service(MockEngine(tok, ""), tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
    raw_key = os.environ.get("STRATA_API_KEY") or cfg.get("api_key", "")
    guard.api_key = "" if explicit_no_key(raw_key) else api_key_of(raw_key)
    guard.allowed_hosts = allowed_hosts_of(cfg.get("allowed_hosts"), os.environ.get("STRATA_ALLOWED_HOSTS", ""))
    guard.cors_origins = origins_of(cfg.get("cors_origins"), "cors_origins", wildcard=True)
    guard.trusted_origins = origins_of(cfg.get("trusted_origins"), "trusted_origins", wildcard=False)
    guard.host_names = host_names_for(host, guard.allowed_hosts, guard.trusted_origins)
    router = ModelRouter(models, engine=engine, max_queue=cfg.get("max_queue", 32))
    try:
        httpd = Server((host, port), make_router_handler(router, guard))
        httpd.guard = guard
    except BaseException:
        router.close()
        raise
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, router


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--host")
    ap.add_argument("--port", type=int)
    ap.add_argument("--engine", choices=("strata", "mock"), default="strata")
    a = ap.parse_args(argv)
    cfg, models = read_models(a.config)
    host = a.host or cfg.get("host", "127.0.0.1")
    port = a.port if a.port is not None else cfg.get("port", 8095)
    if type(port) is not int or not 0 <= port <= 65535:
        ap.error("port must be a whole number in 0..65535")
    finished = threading.Event()
    def request_stop(reason):
        if threading.current_thread() is threading.main_thread():
            raise KeyboardInterrupt
        import _thread
        _thread.interrupt_main()

    lifeline.install_stop_handlers(request_stop, finished)
    httpd = router = None
    try:
        httpd, router = start_router(cfg, models, host=host, port=port, engine=a.engine)
        print(f"[strata router] http://{host}:{httpd.server_port}; models: {', '.join(models)}; loading on demand", flush=True)
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        if httpd is not None:
            httpd.shutdown()
            with httpd.guard.status_lock:
                for cancel in httpd.guard.active_cancels:
                    cancel.set()
        if router is not None:
            router.close()
        if httpd is not None:
            httpd.server_close()
        finished.set()
    return 0


if __name__ == "__main__":
    sys.exit(main())

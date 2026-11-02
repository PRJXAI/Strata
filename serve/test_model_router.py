"""HTTP and process tests for on-demand model selection; no GPU or weights required."""
import concurrent.futures
import http.client
import json
import os
import socket
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest import mock

from serve.model_router import ModelRouter, RouterError, read_models, start_router


def wait_for(predicate, timeout=10):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError("timed out waiting for test state")
        time.sleep(0.02)


class ModelRouting(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for name in ("stock", "abliterated"):
            (self.root / f"{name}.json").write_text(json.dumps({"model_name": f"original-{name}"}))
        self.manifest = self.root / "models.json"
        self.manifest.write_text(json.dumps({"models": {name: f"{name}.json" for name in ("stock", "abliterated")}}))
        self.cfg, self.models = read_models(self.manifest)
        self.httpd = self.router = None
        self.addCleanup(self.close)

    def start(self, **cfg):
        with mock.patch.dict(os.environ, {"STRATA_API_KEY": ""}):
            self.httpd, self.router = start_router({**self.cfg, **cfg}, self.models, port=0, engine="mock")
        self.url = f"http://127.0.0.1:{self.httpd.server_port}"

    def close(self):
        if self.httpd:
            self.httpd.shutdown()
            with self.httpd.guard.status_lock:
                for cancel in self.httpd.guard.active_cancels:
                    cancel.set()
        if self.router:
            self.router.close()
        if self.httpd:
            self.httpd.server_close()

    def request(self, path, body=None, **headers):
        raw = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(self.url + path, raw, {"Content-Type": "application/json", **headers})
        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                return response.status, response.read()
        except urllib.error.HTTPError as e:
            return e.code, e.read()

    def chat(self, model, **kw):
        return self.request("/v1/chat/completions", {"model": model, "messages": [{"role": "user", "content": "Hello"}], **kw})

    def test_discovery_is_lazy_and_unknown_models_do_not_load(self):
        self.start()
        status, raw = self.request("/v1/models")
        self.assertEqual(status, 200)
        rows = json.loads(raw)["data"]
        self.assertEqual([r["id"] for r in rows], ["stock", "abliterated"])
        self.assertTrue(all(r["status"]["value"] == "unloaded" for r in rows))
        self.assertEqual(self.chat("missing")[0], 404)
        self.assertEqual(self.chat([])[0], 404)
        self.assertIsNone(self.router.proc)

    def test_switch_reuses_current_model_and_reaps_previous_process(self):
        self.start()
        for name in ("stock", "stock", "abliterated", "stock"):
            previous = self.router.proc
            same = self.router.active == name
            status, raw = self.chat(name, reasoning_effort="medium", max_tokens=64)
            self.assertEqual(status, 200, raw)
            self.assertEqual(json.loads(raw)["model"], name)
            self.assertEqual(self.router.active, name)
            if previous:
                if same:
                    self.assertIs(previous, self.router.proc)
                else:
                    self.assertIsNotNone(previous.poll())
        rows = json.loads(self.request("/v1/models")[1])["data"]
        self.assertEqual([r["id"] for r in rows if r["status"]["value"] == "loaded"], ["stock"])

    def test_sse_and_other_protocols_survive_routing(self):
        self.start()
        status, raw = self.chat("stock", stream=True, max_tokens=64)
        self.assertEqual(status, 200, raw)
        self.assertIn(b'data: [DONE]', raw)
        self.assertIn(b'"model": "stock"', raw)
        for path, req in [
            ("/v1/messages", {"model": "stock", "max_tokens": 64, "messages": [{"role": "user", "content": "Hello"}]}),
            ("/v1/responses", {"model": "stock", "input": "Hello", "max_output_tokens": 64}),
            ("/v1/messages/count_tokens", {"model": "stock", "messages": [{"role": "user", "content": "Hello"}]})]:
            status, raw = self.request(path, req)
            self.assertEqual(status, 200, (path, raw))
            self.assertIsInstance(json.loads(raw), dict)

    def test_a_different_model_waits_until_stream_finishes(self):
        self.start()
        from serve import lifeline
        popen = lifeline.popen

        def delayed(args, **kw):
            return popen([*args, "--script-delay", "0.04"], **kw)

        with mock.patch("serve.model_router.lifeline.popen", side_effect=delayed):
            req = urllib.request.Request(self.url + "/v1/chat/completions", json.dumps({
                "model": "stock", "messages": [{"role": "user", "content": "Hello"}], "stream": True}).encode(),
                {"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=30) as stream:
                self.assertTrue(stream.readline())
                old = self.router.proc
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    next_model = pool.submit(self.chat, "abliterated")
                    wait_for(lambda: len(self.router.waiters) == 1)
                    self.assertEqual(self.router.active, "stock")
                    self.assertIsNone(old.poll())
                    self.assertIn(b"[DONE]", stream.read())
                    self.assertEqual(next_model.result()[0], 200)
            self.assertIsNotNone(old.poll())

    def test_shutdown_ends_active_inference_without_waiting_for_generation(self):
        self.start()
        from serve import lifeline
        popen = lifeline.popen

        def delayed(args, **kw):
            return popen([*args, "--script-delay", "0.2"], **kw)

        with mock.patch("serve.model_router.lifeline.popen", side_effect=delayed):
            req = urllib.request.Request(self.url + "/v1/chat/completions", json.dumps({
                "model": "stock", "messages": [{"role": "user", "content": "Hello"}], "stream": True}).encode(),
                {"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=30) as stream:
                self.assertTrue(stream.readline())
                old = self.router.proc
                started = time.monotonic()
                self.router.close()
                self.assertLess(time.monotonic()-started, 5)
                self.assertIsNotNone(old.poll())

    def test_sampling_thinking_tools_and_uncapped_values_are_forwarded_verbatim(self):
        self.start()
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        captured = []

        class Capture(BaseHTTPRequestHandler):
            def do_POST(inner):
                captured.append(inner.rfile.read(int(inner.headers["Content-Length"])))
                inner.send_response(200)
                inner.end_headers()
                inner.wfile.write(b'{"ok":true}')

            def log_message(inner, *args):
                pass

        backend = ThreadingHTTPServer(("127.0.0.1", 0), Capture)
        threading.Thread(target=backend.serve_forever, daemon=True).start()
        try:
            self.router.port = backend.server_port
            payload = {"model": "stock", "messages": [], "temperature": 1.0, "top_p": 0.95, "top_k": 20,
                       "min_p": 0, "presence_penalty": 0, "repetition_penalty": 1,
                       "reasoning_effort": "medium", "reasoning_budget_tokens": 0, "max_tokens": 0,
                       "tools": [{"type": "function", "function": {"name": "check", "parameters": {"type": "object"}}}]}
            raw = json.dumps(payload, indent=2).encode()
            req = urllib.request.Request(self.url + "/v1/chat/completions", raw, {"Content-Type": "application/json"})
            with mock.patch.object(self.router, "load"):
                with urllib.request.urlopen(req, timeout=5) as response:
                    self.assertEqual(response.status, 200)
                    response.read()
            self.assertEqual(captured, [raw])
        finally:
            backend.shutdown()
            backend.server_close()
            self.router.port = None

    def test_auth_origin_and_host_checks_happen_before_loading(self):
        self.start(api_key="test-key")
        self.assertEqual(self.chat("stock")[0], 401)
        self.assertEqual(self.request("/v1/models")[0], 401)
        self.assertEqual(self.request("/v1/models", Authorization="Bearer test-key")[0], 200)
        self.assertIsNone(self.router.proc)
        self.assertEqual(self.request("/v1/load", {"model": "stock"}, Authorization="Bearer test-key", Origin="https://evil.example")[0], 403)
        self.httpd.guard.api_key = ""
        self.assertEqual(self.request("/v1/models", Host="evil.example")[0], 403)
        self.assertEqual(self.request("/v1/chat/completions", {"model": "stock"}, Origin="https://evil.example")[0], 403)
        self.assertIsNone(self.router.proc)

    def test_explicit_load_unload_and_restart_after_backend_crash(self):
        self.start()
        self.assertEqual(self.request("/v1/load", {"model": "stock"})[0], 200)
        old = self.router.proc
        old.kill()
        old.wait(timeout=5)
        self.assertEqual(self.chat("stock")[0], 200)
        self.assertNotEqual(self.router.proc.pid, old.pid)
        self.assertEqual(self.request("/v1/unload", {"model": "abliterated"})[0], 200)
        self.assertEqual(self.router.active, "stock")
        self.assertEqual(self.request("/v1/unload", {"model": "stock"})[0], 200)
        self.assertIsNone(self.router.proc)

    def test_existing_model_ids_keep_working_without_a_reload(self):
        self.start()
        self.assertEqual(self.chat("stock")[0], 200)
        proc = self.router.proc
        status, raw = self.chat("original-stock")
        self.assertEqual(status, 200, raw)
        self.assertEqual(json.loads(raw)["model"], "original-stock")
        self.assertIs(self.router.proc, proc)

    def test_ambiguous_original_names_are_rejected(self):
        self.models['abliterated'][1]['model_name'] = 'original-stock'
        with self.assertRaisesRegex(ValueError, 'more than one model'):
            ModelRouter(self.models)

    def test_fifo_and_overload_do_not_switch_while_a_turn_is_active(self):
        self.start(max_queue=2)
        cancel = threading.Event()
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            with self.router.turn(cancel):
                one = pool.submit(self.chat, "stock")
                wait_for(lambda: len(self.router.waiters) == 1)
                two = pool.submit(self.chat, "abliterated")
                wait_for(lambda: len(self.router.waiters) == 2)
                self.assertEqual(self.chat("stock")[0], 429)
                self.assertIsNone(self.router.proc)
            self.assertEqual(one.result()[0], 200)
            self.assertEqual(two.result()[0], 200)
        self.assertEqual(self.router.active, "abliterated")

    def test_queued_disconnect_does_not_load_model(self):
        self.start()
        with self.router.turn(threading.Event()):
            sock = socket.create_connection(("127.0.0.1", self.httpd.server_port))
            body = json.dumps({"model": "stock", "messages": []}).encode()
            sock.sendall(f"POST /v1/chat/completions HTTP/1.0\r\nHost: localhost\r\nContent-Length: {len(body)}\r\nContent-Type: application/json\r\n\r\n".encode() + body)
            wait_for(lambda: len(self.router.waiters) == 1)
            sock.close()
            wait_for(lambda: len(self.router.waiters) == 0)
        self.assertIsNone(self.router.proc)

    def test_bad_json_and_body_limit_do_not_load_model(self):
        self.start()
        for raw in (b"[]", b"{bad"):
            conn = http.client.HTTPConnection("127.0.0.1", self.httpd.server_port)
            conn.request("POST", "/v1/chat/completions", raw)
            self.assertEqual(conn.getresponse().status, 400)
            conn.close()
        with mock.patch.dict(os.environ, {"STRATA_MAX_BODY_MIB": "1"}):
            conn = http.client.HTTPConnection("127.0.0.1", self.httpd.server_port)
            conn.request("POST", "/v1/chat/completions", b"", {"Content-Length": str(2 << 20)})
            self.assertEqual(conn.getresponse().status, 413)
            conn.close()
        self.assertIsNone(self.router.proc)

    def test_failed_start_cleans_up_and_a_good_model_still_loads(self):
        self.start()
        with mock.patch("serve.model_router.ROOT", self.root):
            status, raw = self.chat("abliterated")
        self.assertEqual(status, 503, raw)
        self.assertIsNone(self.router.proc)
        self.assertEqual(self.chat("stock")[0], 200)

    def test_manifest_rejects_invalid_models(self):
        for value in ({}, {"models": []}, {"models": {"": "stock.json"}}, {"models": {"stock": []}}):
            self.manifest.write_text(json.dumps(value))
            with self.assertRaises(ValueError):
                read_models(self.manifest)

    def test_close_rejects_waiters_without_loading(self):
        router = ModelRouter(self.models)
        router.closing = True
        with self.assertRaises(RouterError):
            with router.turn(threading.Event()):
                self.fail("closed router admitted a request")
        router.close()


if __name__ == "__main__":
    unittest.main()

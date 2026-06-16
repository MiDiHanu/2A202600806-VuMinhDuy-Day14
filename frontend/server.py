"""
Dashboard server cho Lab 14 — dùng http.server (stdlib, không thêm dependency).

Chạy:  python frontend/server.py    (mặc định http://localhost:8000)

Cung cấp:
  GET  /                    -> dashboard (static/index.html)
  GET  /static/<file>       -> tài nguyên tĩnh
  GET  /api/status          -> trạng thái hệ thống (mode, models, có report chưa)
  GET  /api/summary         -> reports/summary.json
  GET  /api/results         -> reports/benchmark_results.json
  GET  /api/dataset         -> danh sách golden set (rút gọn)
  GET  /api/corpus          -> corpus.jsonl
  POST /api/evaluate        -> chạy live 1 câu hỏi qua pipeline, trả trace
"""
import os
import sys
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(ROOT, "frontend", "static")
sys.path.insert(0, ROOT)
os.chdir(ROOT)

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

PORT = int(os.getenv("DASHBOARD_PORT", "8000"))

_CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".svg": "image/svg+xml",
}


def _read_json(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _read_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # giảm spam log
        pass

    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body, ensure_ascii=False).encode("utf-8")
        elif isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path):
        if not os.path.isfile(path):
            return self._send(404, {"error": "not found"})
        ext = os.path.splitext(path)[1]
        ctype = _CONTENT_TYPES.get(ext, "application/octet-stream")
        with open(path, "rb") as f:
            data = f.read()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    # ------------------------------------------------------------------ GET
    def do_GET(self):
        path = self.path.split("?")[0]

        if path == "/" or path == "/index.html":
            return self._send_file(os.path.join(STATIC_DIR, "index.html"))

        if path.startswith("/static/"):
            safe = os.path.normpath(path[len("/static/"):]).replace("\\", "/")
            if safe.startswith(".."):
                return self._send(403, {"error": "forbidden"})
            return self._send_file(os.path.join(STATIC_DIR, safe))

        if path == "/api/status":
            return self._send(200, self._status())

        if path == "/api/summary":
            data = _read_json("reports/summary.json")
            return self._send(200 if data else 404, data or {"error": "Chưa có report. Chạy 'python main.py' trước."})

        if path == "/api/results":
            data = _read_json("reports/benchmark_results.json")
            return self._send(200 if data is not None else 404, data if data is not None else {"error": "Chưa có results."})

        if path == "/api/dataset":
            ds = _read_jsonl("data/golden_set.jsonl")
            slim = [
                {"id": c["id"], "question": c["question"], "category": c.get("category"),
                 "type": c.get("metadata", {}).get("type"), "difficulty": c.get("metadata", {}).get("difficulty"),
                 "expected_retrieval_ids": c.get("expected_retrieval_ids", [])}
                for c in ds
            ]
            return self._send(200, slim)

        if path == "/api/corpus":
            return self._send(200, _read_jsonl("data/corpus.jsonl"))

        return self._send(404, {"error": "unknown route"})

    # ----------------------------------------------------------------- POST
    def do_POST(self):
        path = self.path.split("?")[0]
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads((raw or b"{}").decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as e:
            return self._send(400, {"error": f"Body phải là JSON UTF-8 hợp lệ: {e}"})

        if path == "/api/evaluate":
            question = (payload.get("question") or "").strip()
            version = payload.get("version") or "V2_Optimized"
            if not question:
                return self._send(400, {"error": "Thiếu 'question'."})
            try:
                from frontend.live_eval import evaluate_live_sync
                result = evaluate_live_sync(question, version)
                return self._send(200, result)
            except Exception as e:
                import traceback
                traceback.print_exc()
                return self._send(500, {"error": str(e)})

        return self._send(404, {"error": "unknown route"})

    def _status(self):
        summary = _read_json("reports/summary.json")
        # Xác định mode mà không nạp corpus.
        key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        return {
            "mode": "ONLINE" if key else "OFFLINE",
            "judge_models": [
                os.getenv("JUDGE_MODEL_A", "gemini-2.5-flash"),
                os.getenv("JUDGE_MODEL_B", "gemini-2.5-flash-lite"),
            ],
            "arbiter_model": os.getenv("ARBITER_MODEL", "gemini-2.5-flash"),
            "has_reports": summary is not None,
            "dataset_size": len(_read_jsonl("data/golden_set.jsonl")),
            "corpus_size": len(_read_jsonl("data/corpus.jsonl")),
            "report_timestamp": (summary or {}).get("metadata", {}).get("timestamp"),
        }


def main():
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print(f"✅ Dashboard chạy tại  http://localhost:{PORT}")
    print("   Ctrl+C để dừng.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n👋 Đã dừng dashboard.")
        server.shutdown()


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    main()

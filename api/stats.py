import json
from http.server import BaseHTTPRequestHandler

max_duration = 60

STATS = {
    "chunk_size":    1024,
    "overlap_ratio": 0.05,
    "top_k":         5,
}


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        payload = json.dumps(STATS).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        pass

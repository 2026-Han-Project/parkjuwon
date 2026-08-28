"""Grafana 알림 수신 확인용 로컬 웹훅 서버.

외부 서비스(Slack·이메일) 없이도 알림이 실제로 전달되는지 눈으로 확인하려고
둔 것이다. 표준 라이브러리만 쓰므로 추가 설치가 필요 없고, 아무 비용도 들지
않는다.

    python grafana/alert_receiver.py           # 기본 포트 9099
    python grafana/alert_receiver.py --port 9200

받은 알림은 콘솔에 요약해 찍고 grafana/alerts_received.log 에 원문(JSON)을
한 줄씩 append 한다. Grafana 컨테이너에서는 host.docker.internal 로 이 서버에
접근한다(docker-compose.yml 의 GRAFANA_ALERT_WEBHOOK_URL 기본값 참고).
"""

import argparse
import json
import sys
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

LOG_PATH = Path(__file__).parent / "alerts_received.log"


class AlertHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length).decode("utf-8", "replace")
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"ok")

        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            print(f"[{stamp}] JSON이 아닌 요청 수신: {raw[:200]}")
            return

        status = payload.get("status", "?")
        alerts = payload.get("alerts", [])
        print(f"\n[{stamp}] 알림 수신 — status={status}, {len(alerts)}건")
        for a in alerts:
            labels = a.get("labels", {})
            ann = a.get("annotations", {})
            print(f"  · [{labels.get('severity', '-')}] {labels.get('alertname', '(제목 없음)')}")
            if ann.get("summary"):
                print(f"    {ann['summary']}")

        with LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"received_at": stamp, "payload": payload}, ensure_ascii=False) + "\n")

    def log_message(self, *args):
        pass  # 기본 액세스 로그는 끄고 위의 요약만 출력한다


def main():
    # Windows 콘솔 기본 코드페이지(cp949)에서 한글·기호가 깨지거나 예외로 죽는 것을 막는다
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    except (AttributeError, OSError):
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=9099)
    args = ap.parse_args()
    print(f"Grafana 알림 수신 대기 중 — http://0.0.0.0:{args.port}/alert")
    print(f"수신 기록: {LOG_PATH}")
    print("종료하려면 Ctrl+C\n")
    try:
        HTTPServer(("0.0.0.0", args.port), AlertHandler).serve_forever()
    except KeyboardInterrupt:
        print("\n종료했습니다.")


if __name__ == "__main__":
    main()

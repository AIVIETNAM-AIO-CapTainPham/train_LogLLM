#!/usr/bin/env bash
# Start Streamlit demo + Cloudflare tunnel (URL public) trong 1 lệnh.
#   bash demo_app/run_demo.sh
# Ctrl+C để tắt cả Streamlit lẫn tunnel.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Tự chọn port TRỐNG (tránh đụng app streamlit khác đang chạy).
if [[ -z "${PORT:-}" ]]; then
    PORT="$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1]); s.close()')"
fi
echo "Port: $PORT"

cleanup() {
    [[ -n "${ST_PID:-}" ]] && kill "$ST_PID" 2>/dev/null || true
    [[ -n "${CF_PID:-}" ]] && kill "$CF_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "▶ Khởi động Streamlit (port $PORT)..."
uv run streamlit run demo_app/app.py \
    --server.port "$PORT" \
    --server.address 127.0.0.1 \
    --server.headless true &
ST_PID=$!

echo "⏳ Chờ Streamlit sẵn sàng..."
for _ in $(seq 1 90); do
    if curl -s -o /dev/null "http://127.0.0.1:$PORT/"; then
        break
    fi
    sleep 1
done

echo "🌩  Mở Cloudflare tunnel → tìm dòng *.trycloudflare.com bên dưới:"
echo "------------------------------------------------------------------"
cloudflared tunnel --url "http://127.0.0.1:$PORT"

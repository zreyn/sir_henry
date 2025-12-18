```
cd client
uv sync --extra dev
export LIVEKIT_URL="ws://<server>:7880"
export LIVEKIT_TOKEN="<token>"
uv run python src/main.py
```
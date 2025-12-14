# sir_henry
This is a fun personal project where I take a Halloween skeleton pirate and have him talk to people when he sees them.

In these early days:
```
cd agent
uv sync
uv run python src/sirhenry.py
```

To install dev dependencies:
```
uv sync --extra dev
```

This version uses Ollama's service locally, so you have to start it manually. 

Install Ollama with `curl -fsSL https://ollama.com/install.sh | sh`

Run `sudo systemctl edit ollama.service` and add:
```
[Service]
Environment="OLLAMA_KEEP_ALIVE=1"
Environment="OLLAMA_HOST=0.0.0.0:11434"
Type=exec
TimeoutStartSec=30
ExecStartPost=/usr/bin/sleep 5
ExecStartPost=/usr/bin/curl -X POST http://localhost:11434/api/generate -d '{"model": "llama3.2:3b", "keep_alive": -1}'
```

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

Download the main LLM model file and stash it in `agent/models/.`
```
wget https://huggingface.co/tensorblock/Llama-3.2-3B-GGUF/resolve/main/Llama-3.2-3B-Q5_K_M.gguf?download=true
```
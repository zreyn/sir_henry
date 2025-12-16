# sir_henry
This is a fun personal project where I take a Halloween skeleton pirate and have him talk to people when he sees them.

On an Ubuntu machine with an NVIDIA GPU:
```
sudo apt update
sudo apt install -y apt-transport-https ca-certificates curl software-properties-common gnupg lsb-release

sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

### Install NVIDIA Container Toolkit (Required for GPU support)
https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo sed -i -e '/experimental/ s/^#//g' /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Verify GPU access in Docker
docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi
```

## Usage

# Start all services
```
docker compose up -d
```

# View logs
```
docker compose logs -f ollama
docker compose logs -f agent
```

# Stop all services
```
docker compose down
```

For a client, start with the livekit CLI: https://docs.livekit.io/home/cli/

Auto-generate token and join room:
```
lk room join \
  --url ws://YOUR_SERVER_IP:7880 \
  --api-key devkey \
  --api-secret secret \
  --room my-room \
  --identity user1
  ```

Generate a token:
```
lk token create \
  --api-key devkey --api-secret secret \
  --join --room testing \
  --identity human-user \
  --valid-for 24h
```

Dispatch an agent manually:
```
lk dispatch create --url ws://localhost:7880 --api-key devkey --api-secret secret --room testing --agent-name voice-agent
```

List the rooms:
```
lk room list --url ws://localhost:7880 --api-key devkey --api-secret secret
```

For development, you can use a docker-compose.override.yml file to mount the agent code into the voice-agent container for quicker iteration:
```
services:
  voice-agent:
    volumes:
      # Mount source code for live editing (overrides copied code in image)
      - ./agent/src:/app/src:ro
      # Keep existing mounts from main compose file
      - ./agent/models:/app/models
      - ./agent/ref:/app/ref:ro
    # Use 'start' mode for development (auto-reloads on file changes in dev mode)
    command: ["uv", "run", "python", "src/main.py", "dev"]
```

Or, you can rebuild the agent like this:
```
docker compose build agent
docker compose up -d agent
```
# Requirements — macOS

Platform setup for developing and running **me-bot** on macOS (Apple Silicon or Intel).

## Prerequisites

| Tool | Version | Purpose |
| --- | --- | --- |
| macOS | 13+ recommended | Host OS |
| Git | 2.x | Clone and version control |
| Python | 3.12+ | App runtime |
| Docker Desktop | Latest stable | Local Ollama + optional services |
| Terminal | zsh (default) | Shell |

## 1. Install system tools

### Git

Xcode Command Line Tools (includes Git):

```bash
xcode-select --install
```

### Python

Use Homebrew or pyenv. Example with Homebrew:

```bash
brew install python@3.12
python3 --version
```

### Docker Desktop

1. Install from [https://www.docker.com/products/docker-desktop/](https://www.docker.com/products/docker-desktop/)
2. Start Docker Desktop and confirm the daemon is running:

```bash
docker info
```

## 2. Ollama

me-bot talks to Ollama over HTTP at `http://localhost:11434`. You can run Ollama in Docker (recommended if you already use `household-llm`) or install the native macOS app.

### Option A — Docker (shared with household-llm)

If `household-llm` is already running its Ollama container:

```bash
cd ~/src/projects/household-llm
docker compose up -d model
docker ps --filter name=household-llm-model
```

Pull models inside the container:

```bash
docker exec -it household-llm-model ollama pull qwen2.5:7b
docker exec -it household-llm-model ollama pull nomic-embed-text
```

Optional shell alias so `ollama` works in zsh:

```bash
echo "alias ollama='docker exec -it household-llm-model ollama'" >> ~/.zshrc
source ~/.zshrc
```

Verify:

```bash
curl http://localhost:11434/api/tags
```

### Option B — Native Ollama app

```bash
brew install --cask ollama
open -a Ollama
ollama pull qwen2.5:7b
```

Native installs keep models under `~/.ollama/`, separate from Docker volumes.

## 3. Python environment

From the me-bot repo root:

```bash
cd ~/src/projects/me-bot
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
```

When `requirements.txt` is added to the repo:

```bash
pip install -r requirements.txt
```

## 4. Verify setup

```bash
python3 --version
docker ps
curl -s http://localhost:11434/api/tags | python3 -m json.tool
```

Expected: Python 3.12+, Docker running, and at least one Ollama model listed.

## Troubleshooting

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `ollama: command not found` | No native install; using Docker only | Use `docker exec household-llm-model ollama ...` or add the alias above |
| `file does not exist` on `ollama pull` | Model name typo | Use `qwen2.5:7b`, not `qwem2.5:7b` |
| `connection refused` on port 11434 | Ollama container not running | `docker compose up -d model` in household-llm |
| Docker command hangs | Docker Desktop not started | Open Docker Desktop and wait until it reports running |

## Notes

- Apple Silicon Macs should use ARM-native Docker images (default for `ollama/ollama:latest`).
- Keep `.env` local; never commit secrets.

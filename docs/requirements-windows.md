# Requirements — Windows

Platform setup for developing and running **me-bot** on Windows 10/11.

## Prerequisites

| Tool | Version | Purpose |
| --- | --- | --- |
| Windows | 10/11 (64-bit) | Host OS |
| Git | 2.x | Clone and version control |
| Python | 3.12+ | App runtime |
| Docker Desktop | Latest stable | Local Ollama + optional services |
| Terminal | PowerShell or Windows Terminal | Shell |

## 1. Install system tools

### Git

Install via winget:

```powershell
winget install --id Git.Git -e
```

Or download from [https://git-scm.com/download/win](https://git-scm.com/download/win).

### Python

```powershell
winget install --id Python.Python.3.12 -e
python --version
```

During install, enable **Add python.exe to PATH**.

### Docker Desktop

1. Install from [https://www.docker.com/products/docker-desktop/](https://www.docker.com/products/docker-desktop/)
2. Enable WSL 2 backend when prompted (recommended)
3. Start Docker Desktop, then verify:

```powershell
docker info
```

## 2. Ollama

me-bot talks to Ollama over HTTP at `http://localhost:11434`.

### Option A — Docker (recommended for parity with macOS / household-llm)

Clone or open `household-llm`, then start the model service:

```powershell
cd C:\path\to\household-llm
docker compose up -d model
docker ps --filter name=household-llm-model
```

Pull models inside the container:

```powershell
docker exec -it household-llm-model ollama pull qwen2.5:7b
docker exec -it household-llm-model ollama pull nomic-embed-text
```

Optional PowerShell function so `ollama` works in your session profile:

```powershell
function ollama { docker exec -it household-llm-model ollama @args }
```

Add that to `$PROFILE` if you want it in every new terminal.

Verify:

```powershell
curl http://localhost:11434/api/tags
```

### Option B — Native Ollama for Windows

1. Install from [https://ollama.com/download/windows](https://ollama.com/download/windows)
2. Pull models:

```powershell
ollama pull qwen2.5:7b
```

Native installs keep models under `%USERPROFILE%\.ollama\`, separate from Docker volumes.

## 3. Python environment

From the me-bot repo root:

```powershell
cd C:\path\to\me-bot
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

If script execution is blocked:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

When `requirements.txt` is added to the repo:

```powershell
pip install -r requirements.txt
```

## 4. Verify setup

```powershell
python --version
docker ps
curl http://localhost:11434/api/tags
```

Expected: Python 3.12+, Docker running, and at least one Ollama model listed.

## Troubleshooting

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `ollama` not recognized | No native install; using Docker only | Use `docker exec household-llm-model ollama ...` |
| `file does not exist` on pull | Model name typo | Use `qwen2.5:7b`, not `qwem2.5:7b` |
| `connection refused` on 11434 | Ollama container not running | `docker compose up -d model` in household-llm |
| Docker errors about WSL | WSL 2 not enabled | Install/enable WSL 2 per Docker Desktop docs |
| `Activate.ps1` blocked | Execution policy | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |

## Notes

- Prefer WSL 2 with Docker Desktop for better compatibility with Linux-based dev workflows.
- Keep `.env` local; never commit secrets.

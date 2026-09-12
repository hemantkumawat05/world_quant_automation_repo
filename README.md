# Alpha Harness

A local app for generating Alphas on [WorldQuant BRAIN](https://platform.worldquantbrain.com). It runs entirely on your machine.

## Requirements

- [Python](https://www.python.org/) 3.14 or newer
- [uv](https://docs.astral.sh/uv/) 0.12 or newer (`python -m pip install uv`)
- [Node.js](https://nodejs.org) 22.12 or newer
- [pnpm](https://pnpm.io) 12 or newer (`npm install -g pnpm`)

## Setup

```bash
git clone <repo-url> alpha-harness
cd alpha-harness

# Install backend dependencies
cd backend && python -m uv sync

# Install frontend dependencies
cd ../frontend && pnpm install
```

## Quick Start (Windows)

You can launch both the backend and frontend at once using:

- Double-click **`start-all.bat`** (or run `.\start-all.bat` in PowerShell/Command Prompt)

Or run them individually:
- Backend: `.\start-backend.bat`
- Frontend: `.\start-frontend.bat`

## Run (Manual)

Start the backend and frontend in two separate terminals:

```bash
# Terminal 1 - Backend
cd backend
python -m uv run uvicorn alpha_harness.main:app --port 8000 --reload
```

```bash
# Terminal 2 - Frontend
cd frontend
pnpm dev
```

Open http://localhost:5173 and sign in with your BRAIN account.

## Local Data

Your login, API keys, simulations, and synced catalog data are stored in `~/.alpha-harness/`. To use another folder, create a `.env` file in the repository root:

```bash
AH_DATA_DIR=/path/to/folder
```










pnpm dev
python -m uv run uvicorn alpha_harness.main:app --port 8000 --reload
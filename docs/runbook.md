# Runbook

Commands for every task, run from the repo root unless noted. All shell commands assume WSL2.

## Environment

- Node: the system Node is v18, which is too old for vitest and the Papiers CLI. Node 22 is installed through nvm: run `nvm use 22`, or put `~/.nvm/versions/node/v22.23.3/bin` first on PATH. Never install globally for the UI, everything is in `src/ui/node_modules`.
- Python: managed by uv (`uv sync`, `uv run ...`). `uv` lives in `~/.local/bin`.
- `.env` (gitignored, never commit or print it): `NVIDIA_API_KEY`, `GEMINI_API_KEY`. `PAPIERS_TOKEN` is set in the shell profile for the Papiers CLI.
- Optional overrides: `GEMINI_FAST_MODEL`, `GEMINI_STRONG_MODEL`, `FAST_MODEL`, `STRONG_MODEL`, `LLM_TIMEOUT_S`.

## The UI

```bash
cd src/ui && npm run dev
```

Open `http://localhost:5173/`. Modes:

- `?mode=mock` (default in development): the scripted 3 minute mock, always tagged mock.
- `?mode=replay&file=/runs/sessions/<id>/events.jsonl`: a recorded session, labelled recorded run. The dev server (and `vite preview`) serves `runs/` at `/runs/`.
- `?mode=live`: connects to the agent server through the Vite proxy (`/ws`, `/preflight` to port 8000).

Presenter keys: 1 and 2 switch lens, Space pauses, Right steps, Left steps back, Esc clears the selection. The top bar has chapters (jump to a beat), auto-pause at chapters, speed, restart, and in live mode Save as replay and Stop.

Judging build (no mock mode, opens `runs/demo.jsonl`, which you must create by copying the chosen recording there):

```bash
cd src/ui && npm run build:judging && npx vite preview
```

## Live sessions

Start the agent server first, then open `?mode=live`. The page shows the pre-flight check and starts the session on Start.

```bash
uv run uvicorn server.app:app --port 8000
```

Start session is the table demo (capped by `MAX_HYPS`). Start recording runs the full loop for up to `RECORD_HYPS` hypotheses (default 8) or until Stop and saves the log as a replay, its last agent log line names the replay URL. New session in the top bar replaces the session through `POST /restart` without restarting the server.

Server settings: `AUTOSTART=1` (start without the UI), `MAX_HYPS` (default 1, set it empty for an uncapped session), `RECORD_HYPS` (8), `MAIN_ITERS` (2), `BRANCH_ITERS` (3), `IDLE_S` (600), `HELP_DEADLINE_S` (45). Each server start is a new session and uses the GPU. If the UI keeps showing "No agent server is reachable at /ws", the server is not running.

## Headless sessions

```bash
uv run python -m agent.run --id <name> --main-iters 1 --branch-iters 3 --idle 5 --directive "1:speed:Push throughput harder than tile tuning alone." --directive "3:precision:Enforce max error under 1e-3 on the standard and adversarial batches."
```

Each `--directive N:lens:text` is sent once N hypotheses have verdicts. Help requests time out to their default (no human). The log is `runs/sessions/<name>/events.jsonl`, exports are next to it. `runs/night.sh` runs three such sessions back to back (detached with `setsid nohup runs/night.sh > runs/night.log 2>&1 &`).

## Measurements

```bash
uv run python -m harness.bench
```

```bash
uv run python -m harness.stability 10
```

```bash
uv run python -m agent.model_bench
```

Do not run tests or builds while a GPU session or measurement is running, it disturbs timing.

## Tests, lint, type checks

```bash
uv run pytest
```

```bash
uv run pytest -m gpu
```

```bash
uv run ruff check src tests
```

```bash
cd src/ui && npm test && npm run lint && npm run typecheck
```

Regenerate the golden fixture only when the agent's event shapes change on purpose, then rerun the UI golden test:

```bash
GOLDEN_UPDATE=1 uv run pytest tests/test_golden.py
```

## Papiers CLI

```bash
papiers auth status
```

```bash
papiers search "FP16 accumulation" --json
```

The CLI is read-only. The account currently holds one conversation (a summary of the Triton matmul tutorial), no notes and no highlights.

## Screenshots

Headless Chrome from Windows can capture the app at any moment of a replay (virtual time fast-forwards the player):

```bash
"/mnt/c/Program Files/Google/Chrome/Application/chrome.exe" --headless=new --disable-gpu --hide-scrollbars --window-size=1920,1080 --virtual-time-budget=42000 --screenshot="C:\\Users\\Joe\\AppData\\Local\\Temp\\shot.png" "http://localhost:4173/?mode=replay&file=%2Fruns%2Fsessions%2Fcandidate1%2Fevents.jsonl"
```

Use `--window-size=1280,720 --force-device-scale-factor=1.5` to check 150% zoom on a 1920x1080 screen.

## Troubleshooting

- Live mode keeps reconnecting: the agent server on port 8000 is not running.
- A replay shows a red banner: the log failed validation, the banner names the line and the missing field.
- Gemini failures fall back to NIM automatically and the agent log says so. NIM models can take minutes per call.
- `pkill -f "vite preview"` from a shell whose own command line contains that text kills the shell, use the PID.

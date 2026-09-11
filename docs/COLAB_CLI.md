# Colab CLI (terminal workflow)

Use [google-colab-cli](https://github.com/googlecolab/google-colab-cli) to provision a T4 VM, run install/verify, and debug from your terminal.

> **Note:** `colab ssh` is documented upstream but not in CLI 0.6.0 yet. Use `colab console -s SESSION` for a remote shell, or `colab exec` for Python.

## One-time auth

Install (if needed):

```bash
uv tool install google-colab-cli
```

Authenticate (browser opens; paste the code back into the terminal):

```bash
colab sessions
```

Or provision a session (also triggers auth):

```bash
colab new -s bench --gpu T4
```

Token is stored at `~/.config/colab-cli/token.json`.

## Automated debug (recommended)

From repo root, after auth:

```bash
chmod +x scripts/run_colab_debug.sh
./scripts/run_colab_debug.sh bench
```

This will:

1. `colab new -s bench --gpu T4`
2. Run `scripts/colab_remote_setup.py` (clone, install, `pip check`, `verify_colab_env.py`)
3. Print `colab status` and export `reports/colab-debug.md`

## Manual commands

```bash
# Provision GPU
colab new -s bench --gpu T4

# Run setup script (local file executes on remote VM)
colab exec -s bench -f scripts/colab_remote_setup.py --timeout 900

# Remote shell (SSH-like)
colab console -s bench

# Interactive Python
colab repl -s bench

# Stop VM
colab stop -s bench
```

## Ephemeral one-liner

```bash
colab run --gpu T4 --keep --timeout 900 scripts/colab_remote_setup.py
```

`--keep` leaves the session running for follow-up `colab exec -s ...`.

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Auth code prompt | Complete OAuth in browser; paste code |
| `cuda: False` in setup | Use `--gpu T4` on `colab new`, not CPU default |
| Exec timeout | Increase: `--timeout 900` |
| No `colab ssh` | Use `colab console` (0.6.0) |

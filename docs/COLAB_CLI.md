# Colab CLI (terminal workflow)

Run Colab from your terminal with [google-colab-cli](https://github.com/googlecolab/google-colab-cli): spin up a T4, install deps, and debug without clicking through the notebook UI.

> **Note:** `colab ssh` is documented upstream but not in CLI 0.6.0. Use `colab console -s SESSION` for a shell, or `colab exec` for one-off Python.

## One-time auth

Install if needed:

```bash
uv tool install google-colab-cli
```

Log in (browser opens, paste the code back):

```bash
colab sessions
```

Or start a session (also triggers auth):

```bash
colab new -s bench --gpu T4
```

Token lands in `~/.config/colab-cli/token.json`.

## Automated debug

From the repo root, after auth:

```bash
chmod +x scripts/run_colab_debug.sh
./scripts/run_colab_debug.sh bench
```

That provisions `bench` on T4, runs `scripts/colab_remote_setup.py` (clone, install, `pip check`, `verify_colab_env.py`), prints `colab status`, and writes `reports/colab-debug.md`.

## Manual commands

```bash
# Provision GPU
colab new -s bench --gpu T4

# Run setup (local script, remote VM)
colab exec -s bench -f scripts/colab_remote_setup.py --timeout 900

# Remote shell
colab console -s bench

# Interactive Python
colab repl -s bench

# Stop VM
colab stop -s bench
```

## One-liner

```bash
colab run --gpu T4 --keep --timeout 900 scripts/colab_remote_setup.py
```

`--keep` leaves the session up for follow-up `colab exec` calls.

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Auth code prompt | Finish OAuth in the browser, paste the code |
| `cuda: False` in setup | Pass `--gpu T4` to `colab new`, not the CPU default |
| Exec timeout | Bump with `--timeout 900` |
| No `colab ssh` | Use `colab console` on 0.6.0 |

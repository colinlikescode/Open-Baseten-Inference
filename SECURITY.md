# Security

## Reporting a problem

Please do not open a public issue for security problems. Email the maintainers (see the
repository page) with what you found and how to reproduce it. You will get a reply within a
few days.

## What ServePilot does with secrets

- `HF_TOKEN` (and the other Hugging Face token variables) is forwarded to engine processes
  so gated models can download. It is redacted from every command ServePilot prints and from
  its logs.
- `servepilot launch` passes your local Hugging Face token to SkyPilot as a secret. The
  printed task shows `***`; the task file on disk (`~/.cache/servepilot/skypilot/`) does hold
  the real value and is written with mode 0600.
- Nothing is sent anywhere other than Hugging Face (model metadata and downloads), your own
  cloud account through SkyPilot when you use `launch`, and the engines on your machines.

## What ServePilot does not do

- It does not enable `trust_remote_code` unless you pass `--trust-remote-code`.
- It does not add authentication to the API it serves. Bind to `127.0.0.1` (the default) or
  put it behind your own proxy if the machine is reachable from outside.
- It never runs shell strings; engine commands are argument lists.
- It only signals processes it started, and checks the process start time before doing so,
  so a reused PID is never killed by mistake.

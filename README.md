# Crypto Bull/Bear Gauge

A Mistral Workflow that tracks how bullish or bearish two macro/crypto voices on X
are about the crypto market, and emails a daily gauge.

Tracked accounts:

- [@HenrikZeberg](https://x.com/HenrikZeberg)
- [@DrProfitCrypto](https://x.com/DrProfitCrypto)

## How it works

1. `analyze_account` — for each account, a durable agent (`mistral-medium-latest` with
   Mistral's built-in `WebSearchTool`) searches the web for the account's most recent
   crypto posts and returns a strict JSON verdict: a stance score from -100 (max bear)
   to +100 (max bull), a conviction score (0-100), a summary, key claims, and sources.
2. `build_gauge` — aggregates the verdicts into one gauge: a conviction-weighted
   overall score, a label (Strong Bear / Bearish / Neutral / Bullish / Strong Bull),
   the needle position (0 = full bear, 100 = full bull), and the disagreement between
   the two analysts.
3. `send_gauge_email` — sends the gauge by email over Gmail SMTP using a Gmail
   App Password provided through workspace secrets (skipped with `dry_run: true`
   or when no recipient is configured).

## Input

```json
{
  "dry_run": false,
  "recipient_email": "you@example.com"
}
```

- `dry_run`: compute the gauge without sending email (default `false`).
- `recipient_email`: where to send the gauge. Falls back to the
  `GAUGE_RECIPIENT_EMAIL` environment variable when omitted.

## Run locally

```bash
uv sync
cp .env.example .env  # then set MISTRAL_API_KEY
uv run pytest
mistral apps dev      # starts the worker; needs Bun + a .env with MISTRAL_API_KEY
```

## Deploy

Managed deployments build from a pushed GitHub commit:

```bash
git push origin main
```

Then create/redeploy a deployment with Dockerfile path
`deploy/docker/Dockerfile.worker` (build context: repository root).

The deployment's service identity cannot use user-scoped OAuth connector
credentials, so email sending uses Gmail SMTP with an **App Password** stored as
a workspace secret:

1. Google Account → Security → 2-Step Verification on → **App passwords** →
   generate one for Mail.
2. Studio → **Secrets Manager** → create a secret named `GMAIL_APP_PASSWORD`
   with the app password.
3. Bind `GMAIL_APP_PASSWORD` to the deployment via secrets bindings, and set
   `GMAIL_SMTP_USER` (the sending address, plain env var) the same way. Or pass
   `recipient_email` in the workflow/schedule input.

## Schedule

Create a daily schedule against the deployment (e.g. 08:00 UTC) from Studio or via
`client.workflows.schedules.schedule_workflow` with input `{}`.

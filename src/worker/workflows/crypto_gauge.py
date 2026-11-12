"""Crypto Bull/Bear Gauge workflow.

Reads the latest crypto-related posts from two X accounts, scores how bullish
or bearish each account is over the short term (next ~3 months) and the long
term (12+ months), aggregates the verdicts into two gauges, and emails the
result.
"""

import asyncio
import json
import os
import smtplib
from datetime import datetime, timedelta, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any

import pydantic
import mistralai.workflows as workflows
import mistralai.workflows.plugins.mistralai as workflows_mistralai
from mistralai.client.models import TextChunk, WebSearchTool

TRACKED_ACCOUNTS = ["HenrikZeberg", "DrProfitCrypto"]

ANALYST_INSTRUCTIONS = """You are a crypto market sentiment analyst.

Today's date is {today} (UTC).

Your task: research the X (Twitter) account @{account} and assess how bullish
or bearish that account currently is on the crypto market (Bitcoin, Ethereum,
alts in general) over two horizons:
- SHORT TERM: roughly the next 3 months.
- LONG TERM: roughly the next 12 months and beyond (their structural view).

Steps:
1. Use web search to find the account's most recent posts about crypto. Useful
   queries: "site:x.com {account}", "{account} bitcoin forecast", "{account} crypto bull bear",
   "from:{account} btc". Prefer posts from the last 7 days; fall back to the
   most recent posts you can find and note their dates.
2. Read what the account is actually saying: targets, accumulation calls,
   crash warnings, altseason calls, hedging language. Separate what they say
   about the coming weeks/months from their broader structural thesis.
3. Score their stance for each horizon.

Return ONLY a JSON object, with no markdown fences and no commentary before or
after, shaped exactly like this:

{{"account": "{account}", "posts_found": 3, "posts": [{{"date": "YYYY-MM-DD or null", "text": "post text", "url": "post url or null"}}], "stance_score_short_term": 42, "conviction_short_term": 70, "stance_score_long_term": -20, "conviction_long_term": 60, "summary": "one paragraph on their current stance across both horizons", "key_claims": ["claim 1", "claim 2"]}}

Field rules:
- stance_score_short_term: integer from -100 (maximally bearish on the next
  ~3 months) to +100 (maximally bullish). 0 means neutral or genuinely mixed.
- stance_score_long_term: integer from -100 (maximally bearish on the 12+
  month outlook) to +100 (maximally bullish). 0 means neutral or mixed.
- conviction_short_term / conviction_long_term: integers 0-100 for how strongly
  and confidently they express each stance (0 = no recent crypto content or
  pure hedging). If a horizon is not addressed at all, set its conviction to
  10 or below and its score to 0.
- posts: up to 5 of the most recent crypto-relevant posts you found, newest
  first. Include enough text to justify the scores.
- key_claims: 2-4 short quotes or paraphrases capturing the stances.
- If you cannot find any recent crypto posts, set posts_found to 0, both
  scores to 0, both convictions to 0, and explain in summary.
"""


class Post(pydantic.BaseModel):
    date: str | None = None
    text: str
    url: str | None = None


class AccountVerdict(pydantic.BaseModel):
    account: str
    posts_found: int = 0
    posts: list[Post] = pydantic.Field(default_factory=list)
    stance_score_short_term: int = 0
    conviction_short_term: int = 0
    stance_score_long_term: int = 0
    conviction_long_term: int = 0
    summary: str = ""
    key_claims: list[str] = pydantic.Field(default_factory=list)


class AccountGauge(pydantic.BaseModel):
    account: str
    short_score: int
    long_score: int
    conviction_short: int
    conviction_long: int
    short_label: str
    long_label: str
    summary: str
    key_claims: list[str]
    sources: list[str]


class HorizonGauge(pydantic.BaseModel):
    score: int
    label: str
    needle_position_pct: int
    disagreement: int


class Gauge(pydantic.BaseModel):
    as_of: str
    short_term: HorizonGauge
    long_term: HorizonGauge
    accounts: list[AccountGauge]
    email_sent: bool = False


class GaugeInput(pydantic.BaseModel):
    dry_run: bool = False
    recipient_email: str | None = None


_analyst_agents: dict[str, workflows_mistralai.Agent] = {}


def _get_analyst_agent(account: str) -> workflows_mistralai.Agent:
    if account not in _analyst_agents:
        _analyst_agents[account] = workflows_mistralai.Agent(
            model="mistral-medium-latest",
            name=f"crypto-analyst-{account.lower()}",
            description=(
                f"Reads recent crypto posts from X account @{account} and scores "
                "their bull/bear stance on the crypto market over the short and "
                "long term."
            ),
            instructions=ANALYST_INSTRUCTIONS.format(
                today=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                account=account,
            ),
            tools=[WebSearchTool()],
        )
    return _analyst_agents[account]


def extract_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"No JSON object found in agent output: {text[:200]!r}")
    return json.loads(cleaned[start : end + 1])


def label_for_score(score: int) -> str:
    if score >= 60:
        return "Strong Bull"
    if score >= 20:
        return "Bullish"
    if score > -20:
        return "Neutral"
    if score > -60:
        return "Bearish"
    return "Strong Bear"


def needle_position_pct(score: int) -> int:
    return max(0, min(100, round((score + 100) / 2)))


def render_gauge_bar(score: int, width: int = 41) -> str:
    marker = round((score + 100) / 200 * (width - 1))
    slots = ["-"] * width
    slots[width // 2] = "|"
    slots[marker] = "O"
    return "BEAR [" + "".join(slots) + "] BULL"


def _horizon_gauge(pairs: list[tuple[int, int]]) -> HorizonGauge:
    """Aggregate (stance_score, conviction) pairs into one horizon gauge."""
    scores = [score for score, _ in pairs]
    weighted = sum(score * conviction for score, conviction in pairs)
    total_conviction = sum(conviction for _, conviction in pairs)
    overall = round(weighted / total_conviction) if total_conviction else 0
    return HorizonGauge(
        score=overall,
        label=label_for_score(overall),
        needle_position_pct=needle_position_pct(overall),
        disagreement=(max(scores) - min(scores)) if scores else 0,
    )


@workflows.activity(start_to_close_timeout=timedelta(minutes=10), retry_policy_max_attempts=2)
async def analyze_account(account: str) -> dict[str, Any]:
    """Search the web for the account's recent crypto posts and score their stance."""
    agent = _get_analyst_agent(account)
    outputs = await workflows_mistralai.Runner.run(
        agent=agent,
        inputs=(
            f"Research the X account @{account} and score their stance on the "
            "crypto market over the short term (next ~3 months) and the long "
            "term (12+ months). Follow your instructions exactly and return "
            "only the JSON object."
        ),
        session=workflows_mistralai.RemoteSession(),
        max_turns=12,
    )
    text = "\n".join(output.text for output in outputs if isinstance(output, TextChunk))
    data = extract_json_object(text)
    data["account"] = account
    verdict = AccountVerdict.model_validate(data)
    return verdict.model_dump(mode="json")


@workflows.activity()
async def build_gauge(verdicts: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-account verdicts into the short- and long-term gauges."""
    parsed = [AccountVerdict.model_validate(v) for v in verdicts]

    accounts = [
        AccountGauge(
            account=v.account,
            short_score=v.stance_score_short_term,
            long_score=v.stance_score_long_term,
            conviction_short=v.conviction_short_term,
            conviction_long=v.conviction_long_term,
            short_label=label_for_score(v.stance_score_short_term),
            long_label=label_for_score(v.stance_score_long_term),
            summary=v.summary,
            key_claims=v.key_claims,
            sources=[p.url for p in v.posts if p.url],
        )
        for v in parsed
    ]

    gauge = Gauge(
        as_of=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        short_term=_horizon_gauge(
            [(v.stance_score_short_term, v.conviction_short_term) for v in parsed]
        ),
        long_term=_horizon_gauge(
            [(v.stance_score_long_term, v.conviction_long_term) for v in parsed]
        ),
        accounts=accounts,
    )
    return gauge.model_dump(mode="json")


@workflows.activity(start_to_close_timeout=timedelta(minutes=2), retry_policy_max_attempts=3)
async def send_gauge_email(
    subject: str,
    body: str,
    html_body: str,
    recipient: str | None = None,
) -> dict[str, Any]:
    """Send the gauge by email over Gmail SMTP."""
    app_password = os.environ.get("GMAIL_APP_PASSWORD")
    to_address = recipient or os.environ.get("GAUGE_RECIPIENT_EMAIL")
    smtp_user = os.environ.get("GMAIL_SMTP_USER") or to_address
    if not to_address:
        return {"sent": False, "reason": "no recipient configured"}
    if not app_password:
        return {
            "sent": False,
            "reason": "Gmail SMTP not configured (set GMAIL_APP_PASSWORD)",
        }

    message = MIMEMultipart("alternative")
    message["Subject"] = subject
    message["From"] = smtp_user
    message["To"] = to_address
    message.attach(MIMEText(body, "plain"))
    message.attach(MIMEText(html_body, "html"))

    def _send() -> None:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(smtp_user, app_password)
            server.send_message(message)

    await asyncio.to_thread(_send)
    return {"sent": True, "to": to_address}


def _html_needle_row(needle_pct: int) -> str:
    return "".join(
        '<span style="color:#e11d48">&#9679;</span>' if abs(i - needle_pct) <= 1 else '<span style="color:#cbd5e1">&#9632;</span>'
        for i in range(0, 101, 2)
    )


def render_email(gauge: Gauge) -> tuple[str, str, str]:
    """Build (subject, text body, html body) for the gauge email."""
    subject = (
        f"Crypto Gauge {gauge.as_of[:10]} - Short: {gauge.short_term.label} "
        f"({gauge.short_term.score:+d}) / Long: {gauge.long_term.label} "
        f"({gauge.long_term.score:+d})"
    )

    lines = [
        f"CRYPTO BULL/BEAR GAUGE - {gauge.as_of}",
        "",
        "SHORT TERM (next ~3 months)",
        render_gauge_bar(gauge.short_term.score),
        f"Short term: {gauge.short_term.score:+d}/100  [{gauge.short_term.label}]"
        f"  (analyst disagreement: {gauge.short_term.disagreement} points)",
        "",
        "LONG TERM (12+ months)",
        render_gauge_bar(gauge.long_term.score),
        f"Long term: {gauge.long_term.score:+d}/100  [{gauge.long_term.label}]"
        f"  (analyst disagreement: {gauge.long_term.disagreement} points)",
        "",
    ]
    for account in gauge.accounts:
        lines += [
            f"@{account.account}: short {account.short_score:+d} [{account.short_label}]"
            f" (conviction {account.conviction_short}%) / long {account.long_score:+d}"
            f" [{account.long_label}] (conviction {account.conviction_long}%)",
            f"  {account.summary}",
        ]
        for claim in account.key_claims:
            lines.append(f"  - {claim}")
        for source in account.sources[:3]:
            lines.append(f"  source: {source}")
        lines.append("")
    lines.append("Sentiment read of public X posts only. Not financial advice.")

    html_blocks = "".join(
        f"<h3>@{a.account}</h3>"
        f"<p>Short: <b>{a.short_score:+d}</b> [{a.short_label}] (conviction "
        f"{a.conviction_short}%) &middot; Long: <b>{a.long_score:+d}</b> "
        f"[{a.long_label}] (conviction {a.conviction_long}%)</p>"
        f"<p>{a.summary}</p>"
        f"<ul>{''.join(f'<li>{claim}</li>' for claim in a.key_claims)}</ul>"
        for a in gauge.accounts
    )
    html_body = (
        f"<html><body style='font-family:sans-serif'>"
        f"<h2>Crypto Bull/Bear Gauge - {gauge.as_of}</h2>"
        f"<h3>Short term (next ~3 months)</h3>"
        f"<p><b>BEAR</b> &nbsp;{_html_needle_row(gauge.short_term.needle_position_pct)}&nbsp; <b>BULL</b></p>"
        f"<p><b>{gauge.short_term.score:+d}/100</b> [{gauge.short_term.label}] &middot; "
        f"analyst disagreement: {gauge.short_term.disagreement} points</p>"
        f"<h3>Long term (12+ months)</h3>"
        f"<p><b>BEAR</b> &nbsp;{_html_needle_row(gauge.long_term.needle_position_pct)}&nbsp; <b>BULL</b></p>"
        f"<p><b>{gauge.long_term.score:+d}/100</b> [{gauge.long_term.label}] &middot; "
        f"analyst disagreement: {gauge.long_term.disagreement} points</p>"
        f"{html_blocks}"
        f"<p style='color:#64748b'>Sentiment read of public X posts only. Not financial advice.</p>"
        f"</body></html>"
    )
    return subject, "\n".join(lines), html_body


@workflows.workflow.define(
    name="crypto-bull-bear-gauge",
    workflow_display_name="Crypto Bull/Bear Gauge",
    workflow_description=(
        "Reads recent crypto posts from X accounts HenrikZeberg and DrProfitCrypto, "
        "scores their bull/bear stance over the short term (~3 months) and the long "
        "term (12+ months), and emails both gauges."
    ),
)
class CryptoBullBearGauge:
    @workflows.workflow.entrypoint
    async def run(self, input: GaugeInput) -> Gauge:
        verdicts = await asyncio.gather(*(analyze_account(account) for account in TRACKED_ACCOUNTS))
        gauge_data = await build_gauge(list(verdicts))
        gauge = Gauge.model_validate(gauge_data)

        if not input.dry_run:
            subject, body, html_body = render_email(gauge)
            sent = await send_gauge_email(subject, body, html_body, input.recipient_email)
            gauge.email_sent = bool(sent.get("sent"))

        return gauge

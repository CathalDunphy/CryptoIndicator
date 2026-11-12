"""Crypto Bull/Bear Gauge workflow.

Reads the latest crypto-related posts from two X accounts, scores how bullish
or bearish each account is, aggregates the verdicts into a single gauge, and
emails the result.
"""

import asyncio
import json
import os
from datetime import datetime, timedelta, timezone
from typing import Any

import pydantic
import mistralai.workflows as workflows
import mistralai.workflows.plugins.mistralai as workflows_mistralai
from mistralai.client.models import TextChunk, WebSearchTool
from mistralai.workflows import Depends
from mistralai.workflows.plugins.mistralai.connectors import (
    ToolCallClient,
    connector,
    uses_connectors,
)

TRACKED_ACCOUNTS = ["HenrikZeberg", "DrProfitCrypto"]

ANALYST_INSTRUCTIONS = """You are a crypto market sentiment analyst.

Today's date is {today} (UTC).

Your task: research the X (Twitter) account @{account} and assess how bullish
or bearish that account currently is on the crypto market (Bitcoin, Ethereum,
alts in general).

Steps:
1. Use web search to find the account's most recent posts about crypto. Useful
   queries: "site:x.com {account}", "{account} bitcoin forecast", "{account} crypto bull bear",
   "from:{account} btc". Prefer posts from the last 7 days; fall back to the
   most recent posts you can find and note their dates.
2. Read what the account is actually saying: targets, accumulation calls,
   crash warnings, altseason calls, hedging language.
3. Score their current stance on the crypto market.

Return ONLY a JSON object, with no markdown fences and no commentary before or
after, shaped exactly like this:

{{"account": "{account}", "posts_found": 3, "posts": [{{"date": "YYYY-MM-DD or null", "text": "post text", "url": "post url or null"}}], "stance_score": 42, "conviction": 70, "summary": "one paragraph on their current stance", "key_claims": ["claim 1", "claim 2"]}}

Field rules:
- stance_score: integer from -100 (maximally bearish on crypto) to +100
  (maximally bullish). 0 means neutral or genuinely mixed.
- conviction: integer 0-100 for how strongly and confidently they express that
  stance right now (0 = no recent crypto content or pure hedging).
- posts: up to 5 of the most recent crypto-relevant posts you found, newest
  first. Include enough text to justify the score.
- key_claims: 2-4 short quotes or paraphrases capturing the stance.
- If you cannot find any recent crypto posts, set posts_found to 0,
  stance_score to 0, conviction to 0, and explain in summary.
"""


gmail_connector = connector("gmail", run_as="deployment", auto_auth=False)


class Post(pydantic.BaseModel):
    date: str | None = None
    text: str
    url: str | None = None


class AccountVerdict(pydantic.BaseModel):
    account: str
    posts_found: int = 0
    posts: list[Post] = pydantic.Field(default_factory=list)
    stance_score: int = 0
    conviction: int = 0
    summary: str = ""
    key_claims: list[str] = pydantic.Field(default_factory=list)


class AccountGauge(pydantic.BaseModel):
    account: str
    score: int
    conviction: int
    label: str
    summary: str
    key_claims: list[str]
    sources: list[str]


class Gauge(pydantic.BaseModel):
    as_of: str
    overall_score: int
    label: str
    needle_position_pct: int
    disagreement: int
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
                "their bull/bear stance on the crypto market."
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


@workflows.activity(start_to_close_timeout=timedelta(minutes=10), retry_policy_max_attempts=2)
async def analyze_account(account: str) -> dict[str, Any]:
    """Search the web for the account's recent crypto posts and score their stance."""
    agent = _get_analyst_agent(account)
    outputs = await workflows_mistralai.Runner.run(
        agent=agent,
        inputs=(
            f"Research the X account @{account} and score their current stance "
            "on the crypto market. Follow your instructions exactly and return "
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
    """Aggregate per-account verdicts into the bull/bear gauge."""
    parsed = [AccountVerdict.model_validate(v) for v in verdicts]
    weighted = sum(v.stance_score * v.conviction for v in parsed)
    total_conviction = sum(v.conviction for v in parsed)
    overall = round(weighted / total_conviction) if total_conviction else 0

    accounts = [
        AccountGauge(
            account=v.account,
            score=v.stance_score,
            conviction=v.conviction,
            label=label_for_score(v.stance_score),
            summary=v.summary,
            key_claims=v.key_claims,
            sources=[p.url for p in v.posts if p.url],
        )
        for v in parsed
    ]
    scores = [a.score for a in accounts]
    gauge = Gauge(
        as_of=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        overall_score=overall,
        label=label_for_score(overall),
        needle_position_pct=needle_position_pct(overall),
        disagreement=(max(scores) - min(scores)) if scores else 0,
        accounts=accounts,
    )
    return gauge.model_dump(mode="json")


@workflows.activity(start_to_close_timeout=timedelta(minutes=2), retry_policy_max_attempts=3)
async def send_gauge_email(
    subject: str,
    body: str,
    html_body: str,
    recipient: str | None = None,
    gmail: ToolCallClient = Depends(gmail_connector),
) -> dict[str, Any]:
    """Send the gauge by email through the Gmail connector."""
    to_address = recipient or os.environ.get("GAUGE_RECIPIENT_EMAIL")
    if not to_address:
        return {"sent": False, "reason": "no recipient configured"}

    result = await gmail.call_tool(
        tool_name="send_email",
        arguments={
            "to": [to_address],
            "subject": subject,
            "body": body,
            "htmlBody": html_body,
        },
    )
    return {"sent": True, "result": str(result)}


def render_email(gauge: Gauge) -> tuple[str, str, str]:
    """Build (subject, text body, html body) for the gauge email."""
    subject = f"Crypto Bull/Bear Gauge {gauge.as_of[:10]} - {gauge.label} ({gauge.overall_score:+d})"

    lines = [
        f"CRYPTO BULL/BEAR GAUGE - {gauge.as_of}",
        "",
        render_gauge_bar(gauge.overall_score),
        f"Overall: {gauge.overall_score:+d}/100  [{gauge.label}]",
        f"Analyst disagreement: {gauge.disagreement} points",
        "",
    ]
    for account in gauge.accounts:
        lines += [
            f"@{account.account}: {account.score:+d}/100  [{account.label}]  (conviction {account.conviction}%)",
            f"  {account.summary}",
        ]
        for claim in account.key_claims:
            lines.append(f"  - {claim}")
        for source in account.sources[:3]:
            lines.append(f"  source: {source}")
        lines.append("")
    lines.append("Sentiment read of public X posts only. Not financial advice.")

    marker_pct = gauge.needle_position_pct
    row = "".join(
        '<span style="color:#e11d48">&#9679;</span>' if abs(i - marker_pct) <= 1 else '<span style="color:#cbd5e1">&#9632;</span>'
        for i in range(0, 101, 2)
    )
    html_blocks = "".join(
        f"<h3>@{a.account}: {a.score:+d}/100 [{a.label}] (conviction {a.conviction}%)</h3>"
        f"<p>{a.summary}</p>"
        f"<ul>{''.join(f'<li>{claim}</li>' for claim in a.key_claims)}</ul>"
        for a in gauge.accounts
    )
    html_body = (
        f"<html><body style='font-family:sans-serif'>"
        f"<h2>Crypto Bull/Bear Gauge - {gauge.as_of}</h2>"
        f"<p><b>BEAR</b> &nbsp;{row}&nbsp; <b>BULL</b></p>"
        f"<p>Overall: <b>{gauge.overall_score:+d}/100</b> [{gauge.label}] &middot; "
        f"analyst disagreement: {gauge.disagreement} points</p>"
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
        "scores their bull/bear stance, and emails a combined gauge."
    ),
)
@uses_connectors(gmail_connector)
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

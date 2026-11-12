"""Tests for the crypto bull/bear gauge workflow."""

import json
from unittest.mock import AsyncMock, patch

import pytest

import mistralai.workflows.testing as testing
from worker.workflows.crypto_gauge import (
    AccountVerdict,
    CryptoBullBearGauge,
    GaugeInput,
    analyze_account,
    build_gauge,
    extract_json_object,
    label_for_score,
    needle_position_pct,
    render_email,
    render_gauge_bar,
)


def _verdict(account: str, score: int, conviction: int, summary: str = "stance") -> dict:
    return AccountVerdict(
        account=account,
        posts_found=2,
        stance_score=score,
        conviction=conviction,
        summary=summary,
        key_claims=["claim"],
        posts=[{"text": "post", "url": f"https://x.com/{account}/status/1"}],
    ).model_dump(mode="json")


def test_label_for_score_buckets():
    assert label_for_score(80) == "Strong Bull"
    assert label_for_score(30) == "Bullish"
    assert label_for_score(0) == "Neutral"
    assert label_for_score(-30) == "Bearish"
    assert label_for_score(-80) == "Strong Bear"


def test_needle_and_bar_positions():
    assert needle_position_pct(0) == 50
    assert needle_position_pct(-100) == 0
    assert needle_position_pct(100) == 100
    bar = render_gauge_bar(0)
    assert bar.startswith("BEAR [") and bar.endswith("] BULL")
    assert "O" in bar


def test_extract_json_object_handles_fences_and_noise():
    payload = {"account": "x", "stance_score": 10}
    assert extract_json_object(f"```json\n{json.dumps(payload)}\n```") == payload
    assert extract_json_object(f"Here you go:\n{json.dumps(payload)}\nDone.") == payload
    with pytest.raises(ValueError):
        extract_json_object("no json here")


async def test_build_gauge_aggregates_by_conviction():
    verdicts = [_verdict("HenrikZeberg", 80, 90), _verdict("DrProfitCrypto", -40, 10)]
    gauge = await build_gauge(verdicts)
    expected = round((80 * 90 + -40 * 10) / 100)
    assert gauge["overall_score"] == expected
    assert gauge["disagreement"] == 120
    assert gauge["label"] == label_for_score(expected)
    assert gauge["accounts"][0]["sources"] == ["https://x.com/HenrikZeberg/status/1"]


async def test_build_gauge_zero_conviction_is_neutral():
    gauge = await build_gauge([_verdict("HenrikZeberg", 90, 0)])
    assert gauge["overall_score"] == 0
    assert gauge["label"] == "Neutral"


@pytest.fixture
def sample_gauge():
    return {
        "as_of": "2026-02-10 08:00 UTC",
        "overall_score": 42,
        "label": "Bullish",
        "needle_position_pct": 71,
        "disagreement": 60,
        "accounts": [
            {
                "account": "HenrikZeberg",
                "score": 72,
                "conviction": 80,
                "label": "Strong Bull",
                "summary": "Very bullish on altseason.",
                "key_claims": ["Alt-season is here"],
                "sources": ["https://x.com/HenrikZeberg/status/1"],
            },
            {
                "account": "DrProfitCrypto",
                "score": 12,
                "conviction": 60,
                "label": "Neutral",
                "summary": "Cautiously accumulating.",
                "key_claims": ["Accumulating step by step"],
                "sources": [],
            },
        ],
        "email_sent": False,
    }


async def test_render_email_contains_gauge(sample_gauge):
    from worker.workflows.crypto_gauge import Gauge

    gauge = Gauge.model_validate(sample_gauge)
    subject, body, html = render_email(gauge)
    assert "Bullish" in subject and "+42" in subject
    assert "@HenrikZeberg" in body
    assert "BEAR [" in body
    assert "Not financial advice" in body
    assert "<html" in html and "Bullish" in html


def _agent_json(account: str, score: int, conviction: int) -> str:
    return json.dumps(
        {
            "account": account,
            "posts_found": 1,
            "posts": [{"date": "2026-02-09", "text": "Bullish", "url": None}],
            "stance_score": score,
            "conviction": conviction,
            "summary": "s",
            "key_claims": ["c"],
        }
    )


async def test_workflow_dry_run_end_to_end(temporal_env):
    from mistralai.client.models import TextChunk

    async def fake_runner_run(*, agent, inputs, session=None, max_turns=10):
        account = agent.name.removeprefix("crypto-analyst-")
        score = 70 if account == "henrikzeberg" else -30
        return [TextChunk(text=_agent_json(account, score, 80))]

    with patch("worker.workflows.crypto_gauge.workflows_mistralai.Runner.run", new=AsyncMock(side_effect=fake_runner_run)):
        async with testing.create_test_worker(
            temporal_env,
            workflows=[CryptoBullBearGauge],
            activities=[analyze_account, build_gauge],
        ) as worker:
            handle = await temporal_env.client.start_workflow(
                CryptoBullBearGauge.run,
                GaugeInput(dry_run=True),
                id="crypto-gauge-test-1",
                task_queue="test-task-queue",
            )
            result = await handle.result()

    assert result["overall_score"] == 20
    assert result["email_sent"] is False
    assert {a["account"] for a in result["accounts"]} == {"HenrikZeberg", "DrProfitCrypto"}
    assert result["label"] == label_for_score(20)
    assert result["disagreement"] == 100

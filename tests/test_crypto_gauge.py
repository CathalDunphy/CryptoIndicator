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


def _verdict(
    account: str,
    short_score: int,
    short_conviction: int,
    long_score: int | None = None,
    long_conviction: int | None = None,
    summary: str = "stance",
) -> dict:
    return AccountVerdict(
        account=account,
        posts_found=2,
        stance_score_short_term=short_score,
        conviction_short_term=short_conviction,
        stance_score_long_term=short_score if long_score is None else long_score,
        conviction_long_term=short_conviction if long_conviction is None else long_conviction,
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
    payload = {"account": "x", "stance_score_short_term": 10}
    assert extract_json_object(f"```json\n{json.dumps(payload)}\n```") == payload
    assert extract_json_object(f"Here you go:\n{json.dumps(payload)}\nDone.") == payload
    with pytest.raises(ValueError):
        extract_json_object("no json here")


async def test_build_gauge_aggregates_each_horizon_by_conviction():
    verdicts = [
        _verdict("HenrikZeberg", short_score=80, short_conviction=90, long_score=-40, long_conviction=90),
        _verdict("DrProfitCrypto", short_score=-40, short_conviction=10, long_score=80, long_conviction=10),
    ]
    gauge = await build_gauge(verdicts)
    assert gauge["short_term"]["score"] == round((80 * 90 + -40 * 10) / 100)
    assert gauge["short_term"]["disagreement"] == 120
    assert gauge["long_term"]["score"] == round((-40 * 90 + 80 * 10) / 100)
    assert gauge["long_term"]["label"] == label_for_score(gauge["long_term"]["score"])
    assert gauge["accounts"][0]["sources"] == ["https://x.com/HenrikZeberg/status/1"]


async def test_build_gauge_zero_conviction_is_neutral():
    gauge = await build_gauge([_verdict("HenrikZeberg", 90, 0)])
    assert gauge["short_term"]["score"] == 0
    assert gauge["short_term"]["label"] == "Neutral"
    assert gauge["long_term"]["label"] == "Neutral"


@pytest.fixture
def sample_gauge():
    return {
        "as_of": "2026-02-10 08:00 UTC",
        "short_term": {
            "score": 42,
            "label": "Bullish",
            "needle_position_pct": 71,
            "disagreement": 60,
        },
        "long_term": {
            "score": -65,
            "label": "Strong Bear",
            "needle_position_pct": 18,
            "disagreement": 60,
        },
        "accounts": [
            {
                "account": "HenrikZeberg",
                "short_score": 72,
                "long_score": -70,
                "conviction_short": 80,
                "conviction_long": 90,
                "short_label": "Strong Bull",
                "long_label": "Strong Bear",
                "summary": "Short-term rally, long-term crash.",
                "key_claims": ["Fool's Rally"],
                "sources": ["https://x.com/HenrikZeberg/status/1"],
            },
            {
                "account": "DrProfitCrypto",
                "short_score": 12,
                "long_score": -60,
                "conviction_short": 60,
                "conviction_long": 30,
                "short_label": "Neutral",
                "long_label": "Strong Bear",
                "summary": "Cautiously accumulating.",
                "key_claims": ["Accumulating step by step"],
                "sources": [],
            },
        ],
        "email_sent": False,
    }


async def test_render_email_contains_both_gauges(sample_gauge):
    from worker.workflows.crypto_gauge import Gauge

    gauge = Gauge.model_validate(sample_gauge)
    subject, body, html = render_email(gauge)
    assert "Short: Bullish (+42)" in subject and "Long: Strong Bear (-65)" in subject
    assert "SHORT TERM" in body and "LONG TERM" in body
    assert "@HenrikZeberg" in body
    assert body.count("BEAR [") == 2
    assert "Not financial advice" in body
    assert "<html" in html and "Strong Bear" in html
    assert "Short term (next ~3 months)" in html and "Long term (12+ months)" in html


def _agent_json(account: str, short_score: int, short_conviction: int, long_score: int, long_conviction: int) -> str:
    return json.dumps(
        {
            "account": account,
            "posts_found": 1,
            "posts": [{"date": "2026-02-09", "text": "Bullish", "url": None}],
            "stance_score_short_term": short_score,
            "conviction_short_term": short_conviction,
            "stance_score_long_term": long_score,
            "conviction_long_term": long_conviction,
            "summary": "s",
            "key_claims": ["c"],
        }
    )


async def test_workflow_dry_run_end_to_end(temporal_env):
    from mistralai.client.models import TextChunk

    async def fake_runner_run(*, agent, inputs, session=None, max_turns=10):
        account = agent.name.removeprefix("crypto-analyst-")
        short_score = 70 if account == "henrikzeberg" else -30
        long_score = -50 if account == "henrikzeberg" else 40
        return [
            TextChunk(
                text=_agent_json(
                    account,
                    short_score,
                    80,
                    long_score,
                    60,
                )
            )
        ]

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

    expected_short = round((70 * 80 + -30 * 80) / 160)
    expected_long = round((-50 * 60 + 40 * 60) / 120)
    assert result["short_term"]["score"] == expected_short
    assert result["long_term"]["score"] == expected_long
    assert result["short_term"]["disagreement"] == 100
    assert result["long_term"]["disagreement"] == 90
    assert result["email_sent"] is False
    assert {a["account"] for a in result["accounts"]} == {"HenrikZeberg", "DrProfitCrypto"}

"""Expiring Codex reset allowance and startup redemption."""

import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import pytest

from vera_engine import capacity

NOW = datetime(2026, 10, 28, tzinfo=UTC)


def credit(*, hours=24, **changes):
    return (
        dict(
            id="reset-1",
            status="available",
            reset_type="codex_rate_limits",
            is_supported_by_plan=True,
            expires_at=(NOW + timedelta(hours=hours)).isoformat(),
        )
        | changes
    )


def usage(*, remaining=63, credits=None):
    return {
        "plan_type": "prolite",
        "rate_limit": {
            "allowed": remaining > 0,
            "limit_reached": remaining == 0,
            "primary_window": {
                "used_percent": 100 - remaining,
                "limit_window_seconds": 604800,
                "reset_at": (NOW + timedelta(days=3)).timestamp(),
            },
        },
        "rate_limit_reset_credits": {
            "credits": [credit()] if credits is None else credits
        },
    }


def test_expiring_reset_adds_one_full_allowance_and_reduces_pressure():
    ordinary = capacity._parse_codex_usage(usage(credits=[]), now=NOW)
    cap = capacity._parse_codex_usage(
        usage(credits=[credit(), credit(id="reset-2")]), now=NOW
    )
    assert cap.remaining_pct == 163
    assert cap.weekly_remaining_pct == 63
    assert cap.token_used_pct == 37
    assert cap.usage_pressure == pytest.approx(ordinary.usage_pressure / 2)
    assert cap.reset_credit_id == "reset-1"
    assert cap.available


@pytest.mark.parametrize(
    "row",
    [
        credit(hours=48),
        credit(hours=49),
        credit(hours=0),
        credit(hours=-1),
        credit(expires_at=None),
        credit(expires_at="invalid"),
        credit(status="redeemed"),
        credit(is_supported_by_plan=False),
        credit(reset_type="unknown"),
        credit(expires_at="2026-10-29T00:00:00"),
    ],
)
def test_ineligible_resets_do_not_add_quota(row):
    cap = capacity._parse_codex_usage(usage(credits=[row]), now=NOW)
    assert cap.remaining_pct == 63
    assert cap.reset_credit_id is None


def test_earliest_expiring_reset_is_selected():
    cap = capacity._parse_codex_usage(
        usage(credits=[credit(hours=30), credit(hours=1, id="earliest")]), now=NOW
    )
    assert cap.reset_credit_id == "earliest"


def test_exhausted_subscription_remains_selectable_for_startup_redemption():
    cap = capacity._parse_codex_usage(usage(remaining=0), now=NOW)
    assert cap.available
    assert cap.remaining_pct == 100
    assert capacity._prefer_subscription(cap)


class Response:
    def __init__(self, data):
        self.data = data

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return json.dumps(self.data).encode()


def test_usage_fetches_reset_details_without_redeeming():
    payload = usage()
    details = payload["rate_limit_reset_credits"]
    payload["rate_limit_reset_credits"] = {"available_count": 2}
    with (
        patch.object(
            capacity, "_codex_chatgpt_auth", return_value=("token", "account")
        ),
        patch.object(
            capacity, "_urlopen", side_effect=[Response(payload), Response(details)]
        ) as send,
    ):
        capacity._check_codex_usage()
    assert [call.args[0].get_method() for call in send.call_args_list] == ["GET", "GET"]
    assert send.call_args_list[1].args[0].full_url == capacity._CODEX_RESET_CREDITS_URL


@pytest.mark.parametrize(
    "remaining, redeem", [(0, True), (4.9, True), (5, False), (63, False)]
)
def test_startup_uses_raw_weekly_quota_and_deletes_cache(
    tmp_path, monkeypatch, remaining, redeem
):
    cache = tmp_path / "capacity.json"
    cache.write_text("stale")
    monkeypatch.setattr(capacity, "_CAPACITY_CACHE_PATH", cache)
    cap = capacity._parse_codex_usage(usage(remaining=remaining), now=NOW)
    with (
        patch.object(
            capacity, "_codex_chatgpt_auth", return_value=("token", "account")
        ),
        patch.object(capacity, "_check_codex_usage", return_value=cap) as probe,
        patch.object(
            capacity, "_urlopen", return_value=Response({"outcome": "reset"})
        ) as send,
    ):
        capacity.redeem_codex_reset_before_start({"CODEX_HOME": str(tmp_path)})
    probe.assert_called_once_with(auth=("token", "account"))
    assert send.called == redeem
    assert cache.exists() != redeem
    if redeem:
        req = send.call_args.args[0]
        assert req.get_method() == "POST"
        assert json.loads(req.data)["credit_id"] == "reset-1"
        assert json.loads(req.data)["redeem_request_id"]


def test_reset_failure_invalidates_cache_and_stops_startup(tmp_path, monkeypatch):
    cache = tmp_path / "capacity.json"
    cache.write_text("stale")
    monkeypatch.setattr(capacity, "_CAPACITY_CACHE_PATH", cache)
    cap = capacity._parse_codex_usage(usage(remaining=1), now=NOW)
    with (
        patch.object(
            capacity, "_codex_chatgpt_auth", return_value=("token", "account")
        ),
        patch.object(capacity, "_check_codex_usage", return_value=cap),
        patch.object(capacity, "_urlopen", side_effect=TimeoutError),
        pytest.raises(TimeoutError),
    ):
        capacity.redeem_codex_reset_before_start({"CODEX_HOME": str(tmp_path)})
    assert not cache.exists()


def test_reset_metadata_survives_cache_roundtrip_and_bounds_expiry(
    tmp_path, monkeypatch
):
    cache = tmp_path / "capacity.json"
    monkeypatch.setattr(capacity, "_CAPACITY_CACHE_PATH", cache)
    now = NOW.timestamp()
    cap = replace(
        capacity._parse_codex_usage(usage(), now=NOW), reset_credit_expires_at=now + 10
    )
    with patch.object(capacity.time, "time", return_value=now):
        capacity._save_cached_capacities({"openai": cap})
        assert capacity._load_cached_capacities() == {"openai": cap}
    with patch.object(capacity.time, "time", return_value=now + 10):
        assert capacity._load_cached_capacities() is None


@pytest.mark.parametrize("api_key", [False, True])
def test_local_startup_redeems_only_for_subscription_runs(tmp_path, api_key):
    from vera_engine.credentials import CredentialBundle
    from vera_engine.invocation import EngineInvocation
    from vera_engine.render.local import render_local

    invocation = EngineInvocation(
        engine="codex",
        argv=("true",),
        workdir=tmp_path,
        env={"OPENAI_API_KEY": "key"} if api_key else {},
        home_strategy="real",
        timeout_seconds=5,
    )
    with patch.object(capacity, "redeem_codex_reset_before_start") as redeem:
        result = render_local(invocation, CredentialBundle(values={}))
    assert result.returncode == 0
    assert redeem.called != api_key


def test_startup_does_not_redeem_without_an_expiring_reset(tmp_path):
    cap = capacity._parse_codex_usage(usage(remaining=1, credits=[]), now=NOW)
    with (
        patch.object(
            capacity, "_codex_chatgpt_auth", return_value=("token", "account")
        ),
        patch.object(capacity, "_check_codex_usage", return_value=cap),
        patch.object(capacity, "_urlopen") as send,
    ):
        capacity.redeem_codex_reset_before_start({"CODEX_HOME": str(tmp_path)})
    send.assert_not_called()

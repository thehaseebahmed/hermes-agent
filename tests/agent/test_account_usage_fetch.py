import concurrent.futures
import contextvars
import threading
import time
from datetime import datetime, timezone

import pytest

from agent.account_usage import (
    AccountUsageSnapshot,
    AccountUsageWindow,
    _fetch_portal_account,
    fetch_account_usage,
    render_account_usage_lines,
)
from agent.billing_usage import fetch_nous_account as _billing_fetch_nous_account
from providers.base import ProviderProfile


class _UsageProfile(ProviderProfile):
    def __init__(self, snapshot=None, error=None, name="plugin-usage"):
        super().__init__(name=name)
        self.snapshot = snapshot
        self.error = error
        self.calls = 0

    def fetch_account_usage(self, *, base_url=None, api_key=None):
        self.calls += 1
        if self.error:
            raise self.error
        return self.snapshot


class _Response:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


class _Client:
    def __init__(self, payload):
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def get(self, url, headers=None):
        return _Response(self._payload)


class _RoutingClient:
    def __init__(self, payloads, status_code=200):
        self._payloads = payloads
        self._status_code = status_code

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def get(self, url, headers=None):
        return _Response(self._payloads[url], status_code=self._status_code)


def test_fetch_account_usage_codex(monkeypatch):
    monkeypatch.setattr(
        "agent.account_usage.resolve_codex_runtime_credentials",
        lambda refresh_if_expiring=True: {
            "provider": "openai-codex",
            "base_url": "https://chatgpt.com/backend-api/codex",
            "api_key": "access-token",
        },
    )
    monkeypatch.setattr(
        "agent.account_usage._read_codex_tokens",
        lambda: {"tokens": {"account_id": "acct_123"}},
    )
    monkeypatch.setattr(
        "agent.account_usage.httpx.Client",
        lambda timeout=15.0: _Client(
            {
                "plan_type": "pro",
                "rate_limit": {
                    "primary_window": {
                        "used_percent": 15,
                        "reset_at": 1_900_000_000,
                        "limit_window_seconds": 18000,
                    },
                    "secondary_window": {
                        "used_percent": 40,
                        "reset_at": 1_900_500_000,
                        "limit_window_seconds": 604800,
                    },
                },
                "credits": {"has_credits": True, "balance": 12.5},
            }
        ),
    )

    snapshot = fetch_account_usage("openai-codex")

    assert snapshot is not None
    assert snapshot.plan == "Pro"
    assert len(snapshot.windows) == 2
    assert snapshot.windows[0].label == "Session"
    assert snapshot.windows[0].used_percent == 15.0
    assert snapshot.windows[0].reset_at == datetime.fromtimestamp(1_900_000_000, tz=timezone.utc)
    assert "Credits balance: $12.50" in snapshot.details


def _register_profile(monkeypatch, profile):
    import providers

    monkeypatch.setattr(providers, "_REGISTRY", dict(providers._REGISTRY))
    monkeypatch.setattr(providers, "_ALIASES", dict(providers._ALIASES))
    providers.register_provider(profile)


def test_fetch_account_usage_reaches_registered_plugin_profile_and_fails_open(monkeypatch):
    """A profile registered through the public registry (as a plugin does) feeds /usage; a profile
    without the hook, or one whose hook raises, is indistinguishable from today's empty block."""
    snapshot = AccountUsageSnapshot(
        provider="plugin-usage", source="plugin", fetched_at=datetime.now(timezone.utc),
        details=("Credit: 10/100",),
    )
    profile = _UsageProfile(snapshot)
    _register_profile(monkeypatch, profile)
    _register_profile(monkeypatch, ProviderProfile(name="plugin-silent"))
    _register_profile(monkeypatch, _UsageProfile(error=RuntimeError("nope"), name="plugin-broken"))

    assert fetch_account_usage("plugin-usage", base_url="https://plugin.test", api_key="key") is snapshot
    assert profile.calls == 1
    assert fetch_account_usage("plugin-silent") is None
    assert fetch_account_usage("plugin-broken") is None


def test_fetch_account_usage_prefers_builtin_fetcher_over_profile(monkeypatch):
    builtin = AccountUsageSnapshot(
        provider="openrouter", source="builtin", fetched_at=datetime.now(timezone.utc),
    )
    profile = _UsageProfile(
        AccountUsageSnapshot(provider="openrouter", source="plugin", fetched_at=datetime.now(timezone.utc)),
        name="openrouter",
    )
    monkeypatch.setattr("agent.account_usage._USAGE_FETCHERS", {"openrouter": lambda base_url, api_key: builtin})
    _register_profile(monkeypatch, profile)

    assert fetch_account_usage("openrouter") is builtin
    assert profile.calls == 0


def test_fetch_account_usage_openrouter_uses_limit_remaining_and_ignores_deprecated_rate_limit(monkeypatch):
    monkeypatch.setattr(
        "agent.account_usage.resolve_runtime_provider",
        lambda requested, explicit_base_url=None, explicit_api_key=None: {
            "provider": "openrouter",
            "base_url": "https://openrouter.ai/api/v1",
            "api_key": "sk-test",
        },
    )
    monkeypatch.setattr(
        "agent.account_usage.httpx.Client",
        lambda timeout=10.0: _RoutingClient(
            {
                "https://openrouter.ai/api/v1/credits": {
                    "data": {"total_credits": 300.0, "total_usage": 10.92}
                },
                "https://openrouter.ai/api/v1/key": {
                    "data": {
                        "limit": 100.0,
                        "limit_remaining": 70.0,
                        "limit_reset": "monthly",
                        "usage": 12.5,
                        "usage_daily": 0.5,
                        "usage_weekly": 2.0,
                        "usage_monthly": 8.0,
                        "rate_limit": {"requests": -1, "interval": "10s"},
                    }
                },
            }
        ),
    )

    snapshot = fetch_account_usage("openrouter")

    assert snapshot is not None
    assert snapshot.windows == (
        AccountUsageWindow(
            label="API key quota",
            used_percent=30.0,
            detail="$70.00 of $100.00 remaining • resets monthly",
        ),
    )
    assert "Credits balance: $289.08" in snapshot.details
    assert "API key usage: $12.50 total • $0.50 today • $2.00 this week • $8.00 this month" in snapshot.details
    assert all("-1 requests / 10s" not in line for line in render_account_usage_lines(snapshot))


def test_fetch_account_usage_openrouter_omits_quota_window_when_key_has_no_limit(monkeypatch):
    monkeypatch.setattr(
        "agent.account_usage.resolve_runtime_provider",
        lambda requested, explicit_base_url=None, explicit_api_key=None: {
            "provider": "openrouter",
            "base_url": "https://openrouter.ai/api/v1",
            "api_key": "sk-test",
        },
    )
    monkeypatch.setattr(
        "agent.account_usage.httpx.Client",
        lambda timeout=10.0: _RoutingClient(
            {
                "https://openrouter.ai/api/v1/credits": {
                    "data": {"total_credits": 100.0, "total_usage": 25.5}
                },
                "https://openrouter.ai/api/v1/key": {
                    "data": {
                        "limit": None,
                        "limit_remaining": None,
                        "usage": 25.5,
                        "usage_daily": 1.25,
                        "usage_weekly": 4.5,
                        "usage_monthly": 18.0,
                    }
                },
            }
        ),
    )

    snapshot = fetch_account_usage("openrouter")

    assert snapshot is not None
    assert snapshot.windows == ()
    assert "Credits balance: $74.50" in snapshot.details
    assert "API key usage: $25.50 total • $1.25 today • $4.50 this week • $18.00 this month" in snapshot.details


def _ollama_cloud_runtime(**overrides):
    runtime = {"provider": "ollama-cloud", "base_url": "https://ollama.com/v1", "api_key": "ol-test"}
    runtime.update(overrides)
    return lambda requested, explicit_base_url=None, explicit_api_key=None: runtime


def test_fetch_account_usage_ollama_cloud_maps_legacy_session_and_weekly_windows(monkeypatch):
    """Legacy Ollama Cloud plans publish ``remaining_percent`` per window, the inverse of the
    used percentage every surface renders; both windows carry their own reset."""
    monkeypatch.setattr("agent.account_usage.resolve_runtime_provider", _ollama_cloud_runtime())
    monkeypatch.setattr(
        "agent.account_usage.httpx.Client",
        lambda timeout=10.0: _RoutingClient(
            {
                "https://ollama.com/api/balance": {
                    "included": {
                        "session": {"remaining_percent": 75, "resets_at": "2026-10-01T07:00:00Z"},
                        "weekly": {"remaining_percent": 40, "resets_at": "2026-10-05T00:00:00Z"},
                    },
                    "purchased": {"balance_usd": 25},
                }
            }
        ),
    )

    snapshot = fetch_account_usage("ollama-cloud")

    assert snapshot is not None
    assert snapshot.provider == "ollama-cloud"
    assert snapshot.windows == (
        AccountUsageWindow(
            label="Session", used_percent=25.0,
            reset_at=datetime(2026, 10, 1, 7, 0, tzinfo=timezone.utc), detail="75% remaining",
        ),
        AccountUsageWindow(
            label="Weekly", used_percent=60.0,
            reset_at=datetime(2026, 10, 5, 0, 0, tzinfo=timezone.utc), detail="40% remaining",
        ),
    )
    assert snapshot.details == ("Purchased credits: $25.00",)


def test_fetch_account_usage_ollama_cloud_reports_included_allowance_and_purchased_credits(monkeypatch):
    """Current plans report a monthly credit allowance instead of session/weekly windows: the
    window is the share of ``allowance_usd`` spent, resetting when the period rolls over."""
    monkeypatch.setattr("agent.account_usage.resolve_runtime_provider", _ollama_cloud_runtime())
    monkeypatch.setattr(
        "agent.account_usage.httpx.Client",
        lambda timeout=10.0: _RoutingClient(
            {
                "https://ollama.com/api/balance": {
                    "included": {
                        "balance_usd": 72.5,
                        "allowance_usd": 100,
                        "period": {"from": "2026-09-15T09:30:00Z", "until": "2026-10-15T09:30:00Z"},
                    },
                    "purchased": {"balance_usd": 25},
                }
            }
        ),
    )

    snapshot = fetch_account_usage("ollama-cloud")

    assert snapshot is not None
    assert len(snapshot.windows) == 1
    window = snapshot.windows[0]
    assert window.label == "Included credits"
    assert window.used_percent == pytest.approx(27.5)
    assert window.scope == "account"
    assert window.reset_at == datetime(2026, 10, 15, 9, 30, tzinfo=timezone.utc)
    assert window.detail == "$72.50 of $100.00 remaining"
    assert snapshot.details == ("Purchased credits: $25.00",)
    # The window's detail only renders when it has no reset time, so pin the line the surface
    # actually prints: label and percentage from the allowance window.
    assert "Included credits: 72% remaining (28% used)" in " ".join(render_account_usage_lines(snapshot))


def test_fetch_account_usage_ollama_cloud_without_key_returns_none(monkeypatch):
    """A home with no Ollama Cloud credential reports nothing and spends no request."""
    monkeypatch.setattr("agent.account_usage.resolve_runtime_provider", _ollama_cloud_runtime(api_key=""))

    def _no_fetch(timeout=10.0):
        raise AssertionError("a keyless provider must not reach the usage API")

    monkeypatch.setattr("agent.account_usage.httpx.Client", _no_fetch)

    assert fetch_account_usage("ollama-cloud") is None


def test_fetch_account_usage_ollama_cloud_http_failure_is_swallowed(monkeypatch):
    """A 429 (the balance endpoint allows 10 requests a minute) fails open at the dispatch: /usage
    and the picker keep the last snapshot instead of surfacing a traceback."""
    from agent.account_usage import _fetch_ollama_cloud_account_usage

    monkeypatch.setattr("agent.account_usage.resolve_runtime_provider", _ollama_cloud_runtime())
    monkeypatch.setattr(
        "agent.account_usage.httpx.Client",
        lambda timeout=10.0: _RoutingClient({"https://ollama.com/api/balance": {"included": {}}}, status_code=429),
    )

    # The fetcher itself propagates the failure; the dispatch is the layer that fails open.
    with pytest.raises(RuntimeError):
        _fetch_ollama_cloud_account_usage(None, None)
    assert fetch_account_usage("ollama-cloud") is None


def test_plugin_usage_hook_is_bounded_and_fails_open(monkeypatch):
    """A plugin hook that overruns the shared deadline yields None within deadline+1 s on every surface
    (gateway/TUI call ``fetch_account_usage`` with no bound of their own); built-in fetchers are untouched."""
    import threading
    import time

    from agent import account_usage

    started = threading.Event()

    class _Hang(ProviderProfile):
        def fetch_account_usage(self, *, base_url=None, api_key=None):
            started.set()
            time.sleep(5)
            return AccountUsageSnapshot(provider=self.name, source="late", fetched_at=datetime.now(timezone.utc))

    _register_profile(monkeypatch, _Hang(name="plugin-hang"))
    monkeypatch.setattr(account_usage, "PLUGIN_USAGE_HOOK_DEADLINE_S", 0.3)
    builtin_calls = []
    monkeypatch.setattr(account_usage, "_USAGE_FETCHERS",
                        {"openrouter": lambda base_url, api_key: builtin_calls.append(1)})

    t0 = time.monotonic()
    assert account_usage.fetch_account_usage("plugin-hang") is None
    assert time.monotonic() - t0 < 1.3 and started.is_set()
    account_usage.fetch_account_usage("openrouter")
    assert builtin_calls == [1]


def test_plugin_usage_hook_failure_never_reaches_threading_excepthook(monkeypatch):
    """A raising hook fails open in the caller — it must not die on a worker thread, where
    ``threading.excepthook`` prints a traceback into every ``/usage`` surface."""
    import threading

    from agent import account_usage

    class _Boom(ProviderProfile):
        def fetch_account_usage(self, *, base_url=None, api_key=None):
            raise RuntimeError("boom from plugin")

    _register_profile(monkeypatch, _Boom(name="plugin-boom"))
    monkeypatch.setattr(account_usage, "_USAGE_FETCHERS", {})
    escaped = []
    monkeypatch.setattr(threading, "excepthook", lambda args: escaped.append(args.exc_value))

    assert account_usage.fetch_account_usage("plugin-boom") is None
    for t in threading.enumerate():
        if t is not threading.current_thread() and "account-usage" in t.name:
            t.join(2)
    assert escaped == []


@pytest.mark.parametrize("fetch", [_fetch_portal_account, _billing_fetch_nous_account])
def test_fetch_portal_account_is_wall_clock_bounded(monkeypatch, fetch):
    """A portal that accepts the connection but never answers must release the
    caller at ``timeout``, not when the wedged worker finishes on its own
    (``Executor.__exit__`` used to join it via ``shutdown(wait=True)``) — on the
    /usage path and the /billing path alike (#115982)."""
    release = threading.Event()

    def hanging_portal_fetch(*, force_fresh):
        release.wait(timeout=30)
        return object()

    monkeypatch.setattr(
        "hermes_cli.nous_account.get_nous_portal_account_info", hanging_portal_fetch
    )
    started = time.monotonic()
    try:
        with pytest.raises(concurrent.futures.TimeoutError):
            fetch(timeout=0.5)
    finally:
        release.set()
    assert time.monotonic() - started < 10


def test_fetch_portal_account_returns_value_and_keeps_caller_context(monkeypatch):
    marker = contextvars.ContextVar("portal_fetch_test_marker", default="unset")
    sentinel = object()
    seen = {}

    def probing_portal_fetch(*, force_fresh):
        seen["force_fresh"] = force_fresh
        seen["marker"] = marker.get()
        return sentinel

    monkeypatch.setattr(
        "hermes_cli.nous_account.get_nous_portal_account_info", probing_portal_fetch
    )
    token = marker.set("profile-scope")
    try:
        assert _fetch_portal_account(timeout=5) is sentinel
    finally:
        marker.reset(token)
    assert seen == {"force_fresh": True, "marker": "profile-scope"}

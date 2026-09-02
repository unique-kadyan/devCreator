"""Stage deadlines.

The gap these cover: `runtime.timeouts_s` was declared in config and `timeout_key` was
threaded through the stage decorator onto every Stage, but nothing read either. A story
stage then blocked on one socket for 8+ minutes against a nominal 300s budget, because the
provider's httpx timeout is per-read - a response that trickles bytes resets it forever.
"""
import threading
import time

import pytest

from asa.core.runner import StageTimeout, _deadline


def test_deadline_interrupts_a_blocking_call():
    t0 = time.time()
    with pytest.raises(StageTimeout, match="exceeded its 1s budget"):
        with _deadline(1, "story"):
            time.sleep(10)
    assert time.time() - t0 < 5, "the sleep ran to completion; the alarm never fired"


def test_deadline_does_not_fire_when_work_finishes_in_time():
    with _deadline(5, "story"):
        result = 1 + 1
    assert result == 2


def test_zero_budget_means_unbounded():
    # An unset timeout must not become a zero-second one.
    with _deadline(0, "story"):
        time.sleep(0.05)


def test_alarm_is_cleared_afterwards():
    # A leaked alarm fires during whatever runs next, which is far worse than no timeout.
    with _deadline(1, "story"):
        pass
    time.sleep(1.5)          # would raise here if alarm(0) had not been called


def test_off_main_thread_falls_through_instead_of_raising():
    # signal.signal() raises ValueError off the main thread; the dashboard drives stages
    # from a worker, so the guard has to degrade to unbounded rather than explode.
    outcome = {}

    def work():
        try:
            with _deadline(1, "story"):
                time.sleep(0.05)
            outcome["ok"] = True
        except BaseException as e:                                    # noqa: BLE001
            outcome["error"] = repr(e)

    t = threading.Thread(target=work)
    t.start()
    t.join(10)
    assert outcome.get("ok") is True, outcome


def test_timeout_is_not_catchable_as_a_provider_error():
    """The regression that made the whole mechanism a no-op.

    The deadline first raised TimeoutError, which is an OSError. httpx maps socket errors
    to its own exception types, so the transport re-raised it as ConnectTimeout, the
    OpenRouter client turned that into ProviderError, and the model loop treated it as a
    failing model and moved on. signal.alarm() is one-shot, so the deadline was gone and
    the stage ran unbounded from there.
    """
    import httpx

    from asa.core.errors import ProviderError

    with pytest.raises(StageTimeout):
        with _deadline(1, "story"):
            try:
                time.sleep(10)
            except httpx.HTTPError as e:                  # what httpx's transport does
                raise ProviderError(f"timed out: {e}", provider="test") from e
            except Exception as e:                        # any well-meaning broad handler
                raise ProviderError(f"swallowed: {e}", provider="test") from e


def test_timeout_survives_a_real_blocked_socket():
    # 10.255.255.1 is non-routable, so the connect blocks until something interrupts it.
    import httpx

    t0 = time.time()
    with pytest.raises(StageTimeout):
        with _deadline(1, "story"):
            httpx.get("http://10.255.255.1/", timeout=30.0)
    assert time.time() - t0 < 10, "httpx swallowed the deadline again"

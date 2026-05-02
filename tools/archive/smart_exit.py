"""
smart_exit.py — Reusable smart-exit logic from T1a empirical winner.

Two filters layered on top of the live trader's existing SELL path:

  1. CONSECUTIVE-SELL FILTER (k_consecutive)
     Require K consecutive SELL signals at or above the regime's confidence
     threshold before honoring an exit. K=1 = current behavior. K=2 filters
     single-bar noise. K=3+ harmed return on 60d.

  2. MIN-HOLD FILTER (min_hold_hours)
     Ignore SELL signals during the first N hours after entry. Lets entries
     "settle" before exit logic kicks in. N=0 = current behavior. N=4 added
     ~+1pp on top of k=2 in 60d testing.

Empirical winner on 60d cache (2026-04-26): k_consecutive=2, min_hold_hours=4
  Baseline: +49.34% / 50 trades / 70% WR / 21 MAX_HOLD fires
  T1a:      +55.67% / 44 trades / 68% WR / 18 MAX_HOLD fires
  Delta:    +6.34pp

These filters work independently of the existing shield + max_hold mechanics
and override neither — they apply BEFORE the shield's PnL check, so the shield
still serves as the floor on small exits, and max_hold still serves as the
absolute failsafe.

Decision tree at a SELL signal bar:
  1. Is min_hold passed? (hold_since_entry >= min_hold_hours)
     NO → block this SELL, do not increment consecutive counter (signal
          ignored entirely during settle phase)
     YES → continue
  2. Have we seen enough consecutive SELLs? (consecutive_sells >= k_consecutive)
     NO → block this SELL, but increment consecutive counter
     YES → defer to existing shield + max_hold logic

Usage in backtests / live trader:
  from smart_exit import SmartExitState

  state = SmartExitState(k_consecutive=2, min_hold_hours=4)

  # On position entry:
  state.on_entry()

  # On every signal:
  state.observe_signal(signal, confidence, conf_threshold)

  # Before applying shield/max_hold checks:
  if state.should_block_sell(signal, confidence, conf_threshold,
                              hold_since_entry):
      # Skip this SELL — neither shield nor max_hold evaluated
      pass
  else:
      # Proceed to existing shield + max_hold logic
      ...
"""
from dataclasses import dataclass


@dataclass
class SmartExitState:
    """Per-position state for the T1a smart-exit filters.

    Attributes:
        k_consecutive: Number of consecutive SELL signals required to honor
            an exit. 1 = no filter (current behavior).
        min_hold_hours: Number of hours after entry during which SELL signals
            are ignored entirely. 0 = no filter (current behavior).
        consecutive_sells: Internal counter, reset on every BUY (entry) and
            on every non-SELL signal observed during the held position.
    """
    k_consecutive: int = 1
    min_hold_hours: int = 0
    consecutive_sells: int = 0

    def on_entry(self):
        """Call when a new BUY position is opened. Resets the SELL counter."""
        self.consecutive_sells = 0

    def observe_signal(self, signal: str, confidence: float, conf_threshold: float):
        """Update internal state based on the current bar's signal.

        Call this on every bar while in position, BEFORE checking
        should_block_sell. The counter only increments on qualifying SELL
        signals (above confidence threshold). Any other signal resets it.
        """
        if signal == 'SELL' and confidence >= conf_threshold:
            self.consecutive_sells += 1
        else:
            self.consecutive_sells = 0

    def should_block_sell(self, signal: str, confidence: float,
                          conf_threshold: float, hold_since_entry: int) -> bool:
        """Return True if this SELL signal should be blocked by either filter.

        - Returns True if min_hold not yet reached
        - Returns True if not enough consecutive SELLs accumulated
        - Returns False if signal is not a qualifying SELL anyway
          (caller's existing logic handles that)
        - Returns False if both filters pass — caller should then run
          the existing shield + max_hold checks

        Note: this DOES NOT bypass max_hold. Callers should still check
        max_hold AFTER this returns False, OR check max_hold first as a
        higher-priority override. See decision tree in module docstring.
        """
        if signal != 'SELL' or confidence < conf_threshold:
            return False  # not a SELL we care about

        if hold_since_entry < self.min_hold_hours:
            return True  # min-hold filter blocks

        if self.consecutive_sells < self.k_consecutive:
            return True  # consensus filter blocks

        return False  # both filters passed; defer to shield/max_hold


# Default empirical-winner config from T1a 60d sweep (2026-04-26)
DEFAULT_CONFIG = {
    'enabled': False,         # OFF by default — opt-in only
    'k_consecutive': 2,       # require 2 consecutive SELLs
    'min_hold_hours': 4,      # ignore SELL during first 4 hours
}


def from_config(cfg_dict) -> SmartExitState:
    """Build a SmartExitState from a config dict (or fallback to defaults).

    Expected keys:
      smart_exit:
        enabled: bool
        k_consecutive: int
        min_hold_hours: int

    If the smart_exit block is missing or enabled=False, returns a state
    with k_consecutive=1, min_hold_hours=0 (no-op = current behavior).
    """
    se = (cfg_dict or {}).get('smart_exit', {}) if isinstance(cfg_dict, dict) else {}
    if not se.get('enabled', False):
        return SmartExitState(k_consecutive=1, min_hold_hours=0)
    return SmartExitState(
        k_consecutive=int(se.get('k_consecutive', 2)),
        min_hold_hours=int(se.get('min_hold_hours', 4)),
    )

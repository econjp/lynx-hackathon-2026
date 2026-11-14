"""Causal consolidation/expansion strategy using daily closing prices only.

Every returned position is a decision-date target: information through close t,
execution at close t+1, and first price-change P&L on t+2. No macro data or
strategy-P&L feedback enters this module.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Parameters:
    consolidation_months: int = 3
    reference_months: int = 3
    setup_valid_months: int = 3
    overlap_levels: int = 10
    min_overlap: float = 3.0
    min_half_fraction: float = 0.30
    require_inside: bool = True
    quarter_filter: bool = True
    entry_window: int = 5
    pullback_entry: bool = True
    vol_window: int = 20
    initial_stop_sigma: float = 3.0
    trail_steps: int = 3
    suspend_trailing: bool = True
    cooldown_days: int = 5
    rebalance_days: int = 5
    corr_window: int = 100
    corr_threshold: float = 0.70
    corr_shrinkage: float = 0.50
    correlation_budgeting: bool = True
    divergence_preference: float = 0.25
    divergence_memory: int = 5
    reference_capital: float = 100_000.0
    annual_sleeve_risk: float = 0.04
    max_instrument_notional: float = 2.0
    max_gross_notional: float = 6.0
    annualization: int = 260

    def __post_init__(self) -> None:
        if min(self.consolidation_months, self.reference_months, self.setup_valid_months,
               self.overlap_levels, self.entry_window, self.vol_window, self.trail_steps,
               self.rebalance_days, self.corr_window, self.divergence_memory) < 1:
            raise ValueError("Window lengths must be positive.")
        if not 0 <= self.min_half_fraction <= 0.5:
            raise ValueError("min_half_fraction must be between zero and one half.")
        if not 0 <= self.corr_shrinkage <= 1 or not 0 < self.corr_threshold <= 1:
            raise ValueError("Invalid correlation parameters.")
        if self.initial_stop_sigma <= 0 or self.reference_capital <= 0:
            raise ValueError("Initial stop distance and reference capital must be positive.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TrailingStop:
    """Trade-local, symmetric version of the supplied three-close indicator."""

    direction: int
    stop: float
    steps: int = 3
    suspend: bool = True
    armed: bool = True
    reference: float = np.nan
    run: int = 0
    closes: list[float] = field(default_factory=list)

    def update(self, close: float) -> tuple[bool, str]:
        # Test the stop inherited from yesterday before calculating today's update.
        if self.direction * (close - self.stop) < 0:
            return True, "stop_breach"
        previous = self.closes[-1]
        favorable = self.direction * (close - previous) > 0
        adverse = self.direction * (close - previous) < 0
        self.run = self.run + 1 if favorable else 0
        self.closes.append(close)
        if len(self.closes) > self.steps + 2:
            self.closes.pop(0)
        message = ""
        if self.armed:
            if self.run >= self.steps:
                candidate = self.closes[-self.steps]
                updated = max(self.stop, candidate) if self.direction > 0 else min(self.stop, candidate)
                if updated != self.stop:
                    self.stop = updated
                    message = "trail_update"
            if self.suspend and adverse:
                self.armed = False
                self.reference = previous
                message = "trail_suspended"
        elif self.run >= self.steps:
            candidate = self.closes[-self.steps]
            if self.direction * (candidate - self.reference) > 0:
                self.stop = max(self.stop, candidate) if self.direction > 0 else min(self.stop, candidate)
                self.armed = True
                self.reference = np.nan
                message = "trail_rearmed"
        return False, message


@dataclass
class StrategyResult:
    positions: pd.DataFrame
    diagnostics: pd.DataFrame
    events: pd.DataFrame
    setups: pd.DataFrame
    trades: pd.DataFrame
    trade_ids_at_close: pd.DataFrame
    cost_trade_ids_at_close: pd.DataFrame


def validate_prices(prices: pd.DataFrame) -> None:
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise ValueError("Prices need a DatetimeIndex.")
    if prices.empty or len(prices.columns) == 0:
        raise ValueError("Prices are empty.")
    if not prices.index.is_monotonic_increasing or prices.index.has_duplicates:
        raise ValueError("Dates must be sorted and unique.")
    if prices.columns.has_duplicates:
        raise ValueError("Instrument names must be unique.")
    if not np.isfinite(prices.to_numpy(dtype=float)).all():
        raise ValueError("Prices contain missing or infinite values.")
    if (prices.to_numpy() <= 0).any():
        raise ValueError("This version requires positive prices for notional caps.")


def consolidation_statistics(current: np.ndarray, previous: np.ndarray,
                             p: Parameters) -> dict[str, np.ndarray]:
    hi, lo = current.max(axis=0), current.min(axis=0)
    prev_hi, prev_lo = previous.max(axis=0), previous.min(axis=0)
    width = hi - lo
    crosses = np.zeros(current.shape[1])
    # Exact ties to a level retain the previous nonzero side, as in the Pine script.
    for k in range(1, p.overlap_levels + 1):
        level = lo + width * k / (p.overlap_levels + 1)
        last_side = np.zeros(current.shape[1], dtype=int)
        for row in current:
            side = np.sign(row - level).astype(int)
            crosses += (side != 0) & (last_side != 0) & (side != last_side)
            last_side = np.where(side != 0, side, last_side)
    overlap = crosses / p.overlap_levels
    low_fraction = (current < ((hi + lo) / 2)).mean(axis=0)
    inside = (hi <= prev_hi) & (lo >= prev_lo)
    qualifies = ((width > 0) & (overlap >= p.min_overlap)
                 & (low_fraction >= p.min_half_fraction)
                 & (1 - low_fraction >= p.min_half_fraction))
    if p.require_inside:
        qualifies &= inside
    return dict(upper=hi, lower=lo, previous_upper=prev_hi, previous_lower=prev_lo,
                overlap=overlap, lower_fraction=low_fraction, inside=inside,
                qualifies=qualifies)


def prepare_features(prices: pd.DataFrame, p: Parameters) -> dict[str, Any]:
    """Only trailing windows and completed calendar periods are used."""
    validate_prices(prices)
    v = prices.to_numpy(dtype=float)
    n, m = v.shape
    changes = prices.diff()
    vol = changes.rolling(p.vol_window, min_periods=p.vol_window).std(ddof=1).to_numpy()
    corr = changes.rolling(p.corr_window, min_periods=p.corr_window).corr(pairwise=True)
    correlations = corr.to_numpy().reshape(n, m, m)
    prior_high = prices.shift(1).rolling(p.entry_window, min_periods=p.entry_window).max().to_numpy()
    prior_low = prices.shift(1).rolling(p.entry_window, min_periods=p.entry_window).min().to_numpy()
    month_reference = np.full_like(v, np.nan)
    quarter_upper = np.full_like(v, np.nan)
    quarter_lower = np.full_like(v, np.nan)
    months = prices.index.to_period("M")
    quarters = prices.index.to_period("Q")
    month_start = 0
    blocks: list[tuple[int, int]] = []
    q_closes: list[np.ndarray] = []
    setups_by_day: dict[int, dict[str, np.ndarray]] = {}
    setup_records: list[dict[str, Any]] = []
    last_m = np.full(m, np.nan)
    for t in range(n):
        if t and quarters[t] != quarters[t - 1]:
            q_closes.append(v[t - 1].copy())
        if len(q_closes) >= 2:
            quarter_upper[t] = np.maximum(q_closes[-1], q_closes[-2])
            quarter_lower[t] = np.minimum(q_closes[-1], q_closes[-2])
        if t and months[t] != months[t - 1]:
            last_m = v[t - 1].copy()
            blocks.append((month_start, t))
            month_start = t
            count = p.consolidation_months + p.reference_months
            if len(blocks) >= count:
                current_start = blocks[-p.consolidation_months][0]
                previous_start = blocks[-count][0]
                stats = consolidation_statistics(v[current_start:t], v[previous_start:current_start], p)
                setups_by_day[t] = stats
                for j, instrument in enumerate(prices.columns):
                    setup_records.append(dict(date=prices.index[t], instrument=instrument,
                                              **{k: a[j].item() for k, a in stats.items()}))
        month_reference[t] = last_m
    return dict(vol=vol, corr=correlations, prior_high=prior_high, prior_low=prior_low,
                month_reference=month_reference, quarter_upper=quarter_upper,
                quarter_lower=quarter_lower, setups_by_day=setups_by_day,
                setups=pd.DataFrame(setup_records))


def correlation_groups(directions: np.ndarray, correlation: np.ndarray,
                       p: Parameters) -> list[list[int]]:
    active = list(np.flatnonzero(directions))
    groups: list[list[int]] = []
    remaining = set(active)
    while remaining:
        first = min(remaining)
        remaining.remove(first)
        group = [first]
        queue = [first]
        while queue:
            i = queue.pop()
            if not p.correlation_budgeting:
                continue
            linked = [j for j in sorted(remaining)
                      if directions[i] * directions[j] * correlation[i, j] >= p.corr_threshold]
            for j in linked:
                remaining.remove(j)
                group.append(j)
                queue.append(j)
        groups.append(sorted(group))
    return groups


def allocate_risk(directions: np.ndarray, volatility: np.ndarray, correlation: np.ndarray,
                  prices: np.ndarray, advantage: np.ndarray, p: Parameters) -> np.ndarray:
    """A fixed risk sleeve per independent idea; no cash-equity or P&L targeting."""
    result = np.zeros_like(volatility)
    valid = np.isfinite(volatility) & (volatility > 1e-12)
    directions = np.where(valid, directions, 0)
    correlation = np.nan_to_num(correlation, nan=0.0, posinf=0.0, neginf=0.0)
    correlation = np.clip((correlation + correlation.T) / 2, -1, 1)
    np.fill_diagonal(correlation, 1.0)
    budget = p.reference_capital * p.annual_sleeve_risk / np.sqrt(p.annualization)
    for group in correlation_groups(directions, correlation, p):
        ix = np.asarray(group)
        c = correlation[np.ix_(ix, ix)]
        shrunk = (1 - p.corr_shrinkage) * c + p.corr_shrinkage * np.eye(len(ix))
        risk_units = directions[ix] * (1 + p.divergence_preference * advantage[ix])
        # Use the more conservative estimate so shrinkage cannot relax the group cap.
        variance = max(float(risk_units @ c @ risk_units),
                       float(risk_units @ shrunk @ risk_units), 1e-12)
        result[ix] = budget * risk_units / np.sqrt(variance) / volatility[ix]
    limit = p.reference_capital * p.max_instrument_notional / prices
    result = np.clip(result, -limit, limit)
    gross = float(np.sum(np.abs(result) * prices))
    cap = p.reference_capital * p.max_gross_notional
    if gross > cap:
        result *= cap / gross
    return result


def run_strategy(prices: pd.DataFrame, p: Parameters | None = None, *,
                 diagnostics: bool = True, features: dict[str, Any] | None = None) -> StrategyResult:
    p = p or Parameters()
    f = features if features is not None else prepare_features(prices, p)
    v = prices.to_numpy(dtype=float)
    dates = prices.index
    names = list(prices.columns)
    n, m = v.shape
    target = np.zeros_like(v)
    filled = np.zeros(m)
    trade_ids = np.zeros((n, m), dtype=int)
    cost_ids = np.zeros((n, m), dtype=int)
    current_id = np.zeros(m, dtype=int)
    trade_counter = 0
    stops: list[TrailingStop | None] = [None] * m
    trade_records: list[dict[str, Any]] = []
    trade_by_id: dict[int, dict[str, Any]] = {}
    pending_entries: dict[int, dict[str, Any]] = {}
    pending_exit_reason: dict[int, str] = {}
    setup_upper = np.full(m, np.nan)
    setup_lower = np.full(m, np.nan)
    setup_expiry = np.full(m, np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
    setup_date = [pd.NaT] * m
    taken_long = np.zeros(m, dtype=bool)
    taken_short = np.zeros(m, dtype=bool)
    last_exit = np.full(m, -10_000, dtype=int)
    pivot_low = np.full(m, np.nan)
    pivot_high = np.full(m, np.nan)
    long_adv_until = np.full(m, -1, dtype=int)
    short_adv_until = np.full(m, -1, dtype=int)
    diagnostic_records: list[dict[str, Any]] = []
    event_records: list[dict[str, Any]] = []

    def event(t: int, j: int, kind: str, **kwargs: Any) -> None:
        if diagnostics:
            event_records.append(dict(date=dates[t], instrument=names[j], event=kind,
                                      trade_id=int(current_id[j]), **kwargs))

    for t in range(n):
        # Yesterday's decision executes on today's close. This is independent of file length.
        new_filled = target[t - 1].copy() if t else np.zeros(m)
        for j in range(m):
            old_side, new_side = int(np.sign(filled[j])), int(np.sign(new_filled[j]))
            if old_side and new_side != old_side:
                old_id = int(current_id[j])
                rec = trade_by_id[old_id]
                rec.update(exit_date=dates[t], exit_price=float(v[t, j]),
                           exit_reason=pending_exit_reason.pop(j, "target_flat"), status="closed",
                           holding_days=t - rec["entry_row"])
                cost_ids[t, j] = old_id
                event(t, j, "exit_fill", price=float(v[t, j]))
                current_id[j] = 0
                stops[j] = None
                last_exit[j] = t
            if new_side and new_side != old_side:
                trade_counter += 1
                current_id[j] = trade_counter
                meta = pending_entries.pop(j)
                stop = v[t, j] - new_side * p.initial_stop_sigma * meta["signal_volatility"]
                stops[j] = TrailingStop(new_side, float(stop), p.trail_steps,
                                       p.suspend_trailing, closes=[float(v[t, j])])
                rec = dict(trade_id=trade_counter, instrument=names[j], direction=new_side,
                           entry_signal_date=meta["signal_date"], entry_date=dates[t], entry_row=t,
                           entry_price=float(v[t, j]), initial_quantity=float(new_filled[j]),
                           initial_stop=float(stop), entry_type=meta["entry_type"],
                           setup_date=setup_date[j], range_upper=float(setup_upper[j]),
                           range_lower=float(setup_lower[j]), exit_date=pd.NaT,
                           exit_price=np.nan, exit_reason="", status="open",
                           holding_days=0)
                trade_records.append(rec)
                trade_by_id[trade_counter] = rec
                cost_ids[t, j] = trade_counter
                event(t, j, "entry_fill", price=float(v[t, j]), stop=float(stop))
            elif new_side and new_filled[j] != filled[j]:
                cost_ids[t, j] = current_id[j]
                event(t, j, "resize_fill", quantity=float(new_filled[j]))
            trade_ids[t, j] = current_id[j]
        filled = new_filled

        if t >= 2:
            pivot_low = np.where((v[t - 1] < v[t - 2]) & (v[t - 1] < v[t]), v[t - 1], pivot_low)
            pivot_high = np.where((v[t - 1] > v[t - 2]) & (v[t - 1] > v[t]), v[t - 1], pivot_high)
        correlation = np.nan_to_num(f["corr"][t], nan=0.0)
        np.fill_diagonal(correlation, 1.0)
        low_break = np.isfinite(pivot_low) & (v[t] < pivot_low)
        high_break = np.isfinite(pivot_high) & (v[t] > pivot_high)
        related = (correlation >= p.corr_threshold) & ~np.eye(m, dtype=bool)
        long_adv = np.isfinite(pivot_low) & ~low_break & (related & low_break[None, :]).any(axis=1)
        short_adv = np.isfinite(pivot_high) & ~high_break & (related & high_break[None, :]).any(axis=1)
        long_adv_until = np.where(long_adv, t + p.divergence_memory - 1, long_adv_until)
        short_adv_until = np.where(short_adv, t + p.divergence_memory - 1, short_adv_until)

        for j in range(m):
            if not np.isnat(setup_expiry[j]) and dates[t].to_datetime64() >= setup_expiry[j] and not filled[j]:
                setup_upper[j] = setup_lower[j] = np.nan
                setup_expiry[j] = np.datetime64("NaT", "ns")
                event(t, j, "setup_expired")
        if t in f["setups_by_day"]:
            stats = f["setups_by_day"][t]
            expiry = (dates[t].to_period("M") + p.setup_valid_months).start_time.to_datetime64()
            for j in np.flatnonzero(stats["qualifies"]):
                event(t, j, "consolidation_detected", overlap=float(stats["overlap"][j]))
                if filled[j] == 0:
                    setup_upper[j], setup_lower[j] = stats["upper"][j], stats["lower"][j]
                    setup_expiry[j], setup_date[j] = expiry, dates[t]
                    taken_long[j] = taken_short[j] = False
                    event(t, j, "setup_armed", upper=float(setup_upper[j]), lower=float(setup_lower[j]))

        direction = np.sign(filled).astype(int)
        exit_now = np.zeros(m, dtype=bool)
        for j in np.flatnonzero(filled):
            state = stops[j]
            assert state is not None
            if trade_by_id[int(current_id[j])]["entry_row"] == t:
                continue
            hit, message = state.update(float(v[t, j]))
            if message:
                event(t, j, message, stop=state.stop)
            if hit:
                exit_now[j] = True
                direction[j] = 0
                pending_exit_reason[j] = "trailing_stop" if state.stop != trade_by_id[int(current_id[j])]["initial_stop"] else "initial_stop"
                event(t, j, "exit_signal", stop=state.stop, price=float(v[t, j]))

        new_entry = np.zeros(m, dtype=bool)
        q_hi, q_lo = f["quarter_upper"][t], f["quarter_lower"][t]
        long_bias = (v[t] > q_hi) if p.quarter_filter else np.ones(m, dtype=bool)
        short_bias = (v[t] < q_lo) if p.quarter_filter else np.ones(m, dtype=bool)
        for j in range(m):
            valid = (np.isfinite(setup_upper[j]) and not np.isnat(setup_expiry[j])
                     and dates[t].to_datetime64() < setup_expiry[j]
                     and np.isfinite(f["vol"][t, j]) and f["vol"][t, j] > 1e-12)
            if filled[j] != 0 or exit_now[j] or not valid or t - last_exit[j] < p.cooldown_days:
                continue
            side, kind = 0, ""
            if long_bias[j] and v[t, j] > setup_upper[j] and not taken_long[j]:
                side, kind = 1, "breakout"
            elif short_bias[j] and v[t, j] < setup_lower[j] and not taken_short[j]:
                side, kind = -1, "breakout"
            elif p.pullback_entry and t >= p.entry_window:
                monthly = f["month_reference"][t, j]
                if (taken_long[j] and long_bias[j]
                    and f["prior_low"][t, j] < monthly
                    and v[t, j] > max(monthly, f["prior_high"][t, j])
                    and v[t, j] > setup_upper[j]):
                    side, kind = 1, "pullback"
                elif (taken_short[j] and short_bias[j]
                      and f["prior_high"][t, j] > monthly
                      and v[t, j] < min(monthly, f["prior_low"][t, j])
                      and v[t, j] < setup_lower[j]):
                    side, kind = -1, "pullback"
            if side:
                direction[j] = side
                new_entry[j] = True
                taken_long[j] |= side > 0
                taken_short[j] |= side < 0
                pending_entries[j] = dict(signal_date=dates[t], entry_type=kind,
                                          signal_volatility=float(f["vol"][t, j]))
                event(t, j, "entry_signal", direction=side, entry_type=kind, price=float(v[t, j]))

        if t % p.rebalance_days == 0 or new_entry.any():
            advantage = np.where(direction > 0, t <= long_adv_until, t <= short_adv_until).astype(float)
            target[t] = allocate_risk(direction, f["vol"][t], correlation, v[t], advantage, p)
        else:
            target[t] = np.where(direction != 0, filled, 0.0)
        # An exit always takes precedence over a scheduled rebalance.
        target[t, exit_now] = 0.0
        gross = float(np.sum(np.abs(target[t]) * v[t]))
        if gross > p.reference_capital * p.max_gross_notional:
            target[t] *= p.reference_capital * p.max_gross_notional / gross

        if diagnostics:
            for j in range(m):
                state = stops[j]
                diagnostic_records.append(dict(
                    date=dates[t], instrument=names[j], close=float(v[t, j]),
                    desired_quantity=float(target[t, j]), filled_quantity=float(filled[j]),
                    earning_quantity=float(target[t - 2, j]) if t >= 2 else 0.0,
                    trade_id=int(current_id[j]), volatility=float(f["vol"][t, j]),
                    quarter_upper=float(q_hi[j]), quarter_lower=float(q_lo[j]),
                    month_reference=float(f["month_reference"][t, j]),
                    range_upper=float(setup_upper[j]), range_lower=float(setup_lower[j]),
                    setup_valid=bool(valid_setup(dates[t], setup_upper[j], setup_expiry[j])),
                    stop=state.stop if state else np.nan,
                    trail_reference=state.reference if state else np.nan,
                    trail_armed=state.armed if state else False,
                    favorable_steps=state.run if state else 0,
                    quarter_bias=1 if long_bias[j] else -1 if short_bias[j] else 0,
                    long_divergence=bool(t <= long_adv_until[j]),
                    short_divergence=bool(t <= short_adv_until[j]),
                    correlated_neighbors=int(related[j].sum()),
                    entry_signal=bool(new_entry[j]), exit_signal=bool(exit_now[j])))

    for rec in trade_records:
        if rec["status"] == "open":
            rec.update(holding_days=n - 1 - rec["entry_row"], last_mark_date=dates[-1],
                       last_mark_price=float(v[-1, names.index(rec["instrument"])]))
    index_name = prices.index.name or "AsOfDate"
    positions = pd.DataFrame(target, index=dates, columns=names).rename_axis(index_name)
    return StrategyResult(positions, pd.DataFrame(diagnostic_records), pd.DataFrame(event_records),
                          f["setups"], pd.DataFrame(trade_records),
                          pd.DataFrame(trade_ids, index=dates, columns=names),
                          pd.DataFrame(cost_ids, index=dates, columns=names))


def valid_setup(date: pd.Timestamp, upper: float, expiry: np.datetime64) -> bool:
    return bool(np.isfinite(upper) and not np.isnat(expiry) and date.to_datetime64() < expiry)


def trend_benchmark(prices: pd.DataFrame, p: Parameters) -> pd.DataFrame:
    """50-day trend with the same fixed risk sleeves, weekly sizing and caps.

    It deliberately excludes the supplied example's strategy-P&L risk scaling.
    """
    f = prepare_features(prices, p)
    directions = np.sign(prices.diff(50).fillna(0).to_numpy()).astype(int)
    output = np.zeros_like(prices.to_numpy())
    for t in range(len(prices)):
        changed = t == 0 or np.any(directions[t] != directions[t - 1])
        if changed or t % p.rebalance_days == 0:
            output[t] = allocate_risk(directions[t], f["vol"][t], f["corr"][t],
                                      prices.iloc[t].to_numpy(), np.zeros(prices.shape[1]), p)
        else:
            output[t] = output[t - 1]
        gross = float(np.sum(np.abs(output[t]) * prices.iloc[t].to_numpy()))
        cap = p.reference_capital * p.max_gross_notional
        if gross > cap:
            output[t] *= cap / gross
    return pd.DataFrame(output, index=prices.index, columns=prices.columns)

"""Transparent P&L accounting and diagnostics for the Lynx execution convention."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from strategy import Parameters, StrategyResult


@dataclass
class Evaluation:
    daily: pd.DataFrame
    gross_by_instrument: pd.DataFrame
    costs_by_instrument: pd.DataFrame
    net_by_instrument: pd.DataFrame
    earning_positions: pd.DataFrame
    cost_volatility: pd.DataFrame


def sharpe(values: pd.Series, annualization: int = 260) -> float:
    s = values.dropna()
    std = s.std(ddof=1)
    return float(s.mean() / std * np.sqrt(annualization)) if len(s) >= 2 and std > 1e-15 else np.nan


def adjusted_sharpe(values: pd.Series, annualization: int = 260) -> float:
    sr = sharpe(values, annualization)
    if len(values.dropna()) < 4 or not np.isfinite(sr):
        return np.nan
    return float(sr * (1 + sr / 6 * values.skew() - sr ** 2 / 24 * values.kurt()))


def evaluate(positions: pd.DataFrame, prices: pd.DataFrame, p: Parameters, *,
             cost_coefficient: float = 0.02, lag: int = 2,
             cost_model: str = "lynx_code") -> Evaluation:
    """Reproduce the supplied 2026 evaluator, including its cost-booking lag.

    Positive cost_coefficient is a deduction. For the PDF alternative, use
    cost_model='pdf_constant' and cost_coefficient=0.0002. Shifts apply to trading
    rows, not calendar days. NaNs in the initial accounting rows are preserved.
    """
    if not positions.index.equals(prices.index) or list(positions.columns) != list(prices.columns):
        raise ValueError("Positions and prices must have identical dates and instrument columns.")
    if not np.isfinite(positions.to_numpy()).all():
        raise ValueError("Every decision-date target must be finite; use zero when flat.")
    if lag < 1 or cost_coefficient < 0:
        raise ValueError("Lag must be positive and the cost coefficient nonnegative.")
    changes = prices.diff()
    held = positions.shift(lag)
    trades = held.diff().abs()
    # Match the organizer's approximation, not a strategy feature. All strategy
    # positions are zero during these first few back-filled cost-volatility rows.
    sigma = changes.rolling(100, min_periods=10).std(ddof=1).ffill().shift(1).bfill()
    gross = held * changes
    if cost_model == "lynx_code":
        costs = trades * sigma * cost_coefficient
    elif cost_model == "pdf_constant":
        costs = trades * cost_coefficient
    else:
        raise ValueError(f"Unknown cost model: {cost_model}")
    net = gross - costs
    pnl = net.sum(axis=1, min_count=1)
    earning = held.fillna(0)
    daily = pd.DataFrame(dict(
        net_pnl=pnl,
        gross_pnl=gross.sum(axis=1, min_count=1),
        transaction_cost=costs.sum(axis=1, min_count=1),
        gross_notional=(earning.abs() * prices.shift(1)).sum(axis=1),
        net_notional=(earning * prices.shift(1)).sum(axis=1),
        active_instruments=(earning.abs() > 1e-12).sum(axis=1),
        standalone_daily_vol=(earning.abs() * sigma).sum(axis=1),
        signed_daily_vol=(earning * sigma).sum(axis=1),
        traded_daily_vol=(trades * sigma).sum(axis=1, min_count=1),
    ), index=prices.index)
    daily["gross_leverage"] = daily.gross_notional / p.reference_capital
    daily["net_leverage"] = daily.net_notional / p.reference_capital
    daily["pnl_on_reference_capital"] = pnl / p.reference_capital
    return Evaluation(daily, gross, costs, net, held, sigma)


def curve_and_rolling(daily: pd.DataFrame, p: Parameters,
                      benchmark_pnl: pd.Series | None = None) -> pd.DataFrame:
    """Rebase a period's equity and rolling windows at that period's own start."""
    d = daily.copy()
    net = d.net_pnl
    d["cumulative_net_pnl"] = net.fillna(0).cumsum()
    d["cumulative_gross_pnl"] = d.gross_pnl.fillna(0).cumsum()
    d["cumulative_costs"] = d.transaction_cost.fillna(0).cumsum()
    d["reference_equity"] = p.reference_capital + d.cumulative_net_pnl
    peaks = np.maximum.accumulate(np.r_[p.reference_capital, d.reference_equity.to_numpy()])[1:]
    d["drawdown_units"] = d.reference_equity - peaks
    d["drawdown_reference_pct"] = d.drawdown_units / p.reference_capital * 100
    d["drawdown_peak_pct"] = d.drawdown_units / peaks * 100
    d["underwater_days"] = underwater_lengths(d.drawdown_units.to_numpy())
    for window in (20, 63, 126, 260):
        roll = net.rolling(window, min_periods=window)
        avg, std = roll.mean(), roll.std(ddof=1)
        sr = avg / std.replace(0, np.nan) * np.sqrt(p.annualization)
        d[f"rolling_{window}_pnl"] = roll.sum()
        d[f"rolling_{window}_sharpe"] = sr
        d[f"rolling_{window}_adjusted_sharpe"] = sr * (1 + sr / 6 * roll.skew() - sr ** 2 / 24 * roll.kurt())
        d[f"rolling_{window}_annual_vol_pct"] = std / p.reference_capital * np.sqrt(p.annualization) * 100
        d[f"rolling_{window}_mean_pnl_annualized"] = avg * p.annualization
        d[f"rolling_{window}_winning_day_fraction"] = (net > 0).astype(float).where(net.notna()).rolling(window).mean()
        d[f"rolling_{window}_costs"] = d.transaction_cost.rolling(window).sum()
        d[f"rolling_{window}_holding_days"] = (2 * d.standalone_daily_vol.rolling(window).mean()
                                                / d.traded_daily_vol.rolling(window).mean().replace(0, np.nan))
        d[f"rolling_{window}_position_bias"] = (d.signed_daily_vol.rolling(window).mean()
                                                / d.standalone_daily_vol.rolling(window).mean().replace(0, np.nan))
        # Drawdown within each window includes the window's zero starting P&L.
        d[f"rolling_{window}_max_drawdown_units"] = roll.apply(window_drawdown, raw=True)
        if benchmark_pnl is not None:
            bench = benchmark_pnl.reindex(d.index)
            d[f"rolling_{window}_benchmark_correlation"] = roll.corr(bench)
    return d


def window_drawdown(values: np.ndarray) -> float:
    path = np.r_[0.0, np.cumsum(values)]
    return float(np.min(path - np.maximum.accumulate(path)))


def underwater_lengths(drawdown: np.ndarray) -> np.ndarray:
    output = np.zeros(len(drawdown), dtype=int)
    run = 0
    for i, dd in enumerate(drawdown):
        run = run + 1 if dd < -1e-9 else 0
        output[i] = run
    return output


def performance(ev: Evaluation, p: Parameters, mask: np.ndarray | pd.Series | None = None,
                trades: pd.DataFrame | None = None) -> dict[str, Any]:
    d = ev.daily.loc[mask] if mask is not None else ev.daily
    values = d.net_pnl.dropna()
    if len(values) == 0:
        raise ValueError("No P&L observations in the requested period.")
    annual = p.annualization
    cumulative = values.cumsum()
    path = np.r_[0.0, cumulative.to_numpy()]
    dd = path - np.maximum.accumulate(path)
    equity = p.reference_capital + path
    equity_peaks = np.maximum.accumulate(equity)
    peak_dd = (equity - equity_peaks) / equity_peaks
    neg = np.minimum(values.to_numpy(), 0)
    downside = float(np.sqrt(np.mean(neg ** 2)))
    positive_sum = float(values[values > 0].sum())
    negative_sum = float(-values[values < 0].sum())
    norm_positions = d.standalone_daily_vol
    norm_trades = d.traded_daily_vol
    total_net = float(values.sum())
    var_cut = values.quantile(0.05)
    active = d.active_instruments > 0
    metrics: dict[str, Any] = dict(
        start=str(d.index[0].date()), end=str(d.index[-1].date()),
        rows=len(d), pnl_observations=len(values), years=len(values) / annual,
        net_pnl=total_net, gross_pnl=float(d.gross_pnl.sum()),
        transaction_cost=float(d.transaction_cost.sum()),
        net_sharpe=sharpe(values, annual), adjusted_sharpe=adjusted_sharpe(values, annual),
        gross_sharpe=sharpe(d.gross_pnl, annual),
        skewness=float(values.skew()), kurtosis=float(values.kurt() + 3),
        excess_kurtosis=float(values.kurt()),
        annual_mean_pnl=float(values.mean() * annual),
        annual_vol_pnl=float(values.std(ddof=1) * np.sqrt(annual)),
        annual_return_reference_pct=float(values.mean() * annual / p.reference_capital * 100),
        annual_vol_reference_pct=float(values.std(ddof=1) * np.sqrt(annual) / p.reference_capital * 100),
        total_return_reference_pct=total_net / p.reference_capital * 100,
        reference_equity_end=p.reference_capital + total_net,
        reference_cagr_pct=float(((p.reference_capital + total_net) / p.reference_capital)
                                 ** (annual / len(values)) - 1) * 100
        if p.reference_capital + total_net > 0 else np.nan,
        max_drawdown_units=float(dd.min()),
        max_drawdown_reference_pct=float(dd.min() / p.reference_capital * 100),
        max_drawdown_peak_pct=float(peak_dd.min() * 100),
        longest_underwater_days=int(underwater_lengths(dd).max()),
        sortino=float(values.mean() / downside * np.sqrt(annual)) if downside > 0 else np.nan,
        daily_profit_factor=positive_sum / negative_sum if negative_sum > 0 else np.nan,
        winning_day_fraction=float((values > 0).mean()),
        losing_day_fraction=float((values < 0).mean()),
        zero_pnl_day_fraction=float((values.abs() < 1e-12).mean()),
        invested_day_fraction=float(active.mean()),
        active_day_sharpe=sharpe(d.loc[active, "net_pnl"], annual),
        average_active_instruments=float(d.active_instruments.mean()),
        max_active_instruments=int(d.active_instruments.max()),
        average_gross_leverage=float(d.gross_leverage.mean()),
        max_gross_leverage=float(d.gross_leverage.max()),
        average_net_leverage=float(d.net_leverage.mean()),
        position_bias=float(d.signed_daily_vol.mean() / norm_positions.mean())
        if norm_positions.mean() > 0 else np.nan,
        average_holding_days=float(2 * norm_positions.mean() / norm_trades.mean())
        if norm_trades.mean() > 0 else np.nan,
        worst_day=float(values.min()), best_day=float(values.max()),
        daily_var95_loss=float(max(0.0, -var_cut)),
        daily_expected_shortfall95_loss=float(max(0.0, -values[values <= var_cut].mean())),
    )
    if trades is not None and not trades.empty:
        closed = trades[(trades.entry_date >= d.index[0]) & (trades.exit_date <= d.index[-1])
                        & (trades.status == "closed")]
        entered = trades[(trades.entry_date >= d.index[0]) & (trades.entry_date <= d.index[-1])]
        metrics.update(trades_entered=int(len(entered)), complete_trades=int(len(closed)),
                       complete_trade_win_rate=float((closed.net_pnl > 0).mean()) if len(closed) else np.nan,
                       complete_trade_average_pnl=float(closed.net_pnl.mean()) if len(closed) else np.nan,
                       complete_trade_median_holding=float(closed.holding_days.median()) if len(closed) else np.nan,
                       open_trades_at_end=int(((trades.entry_date <= d.index[-1])
                                               & (trades.exit_date.isna() | (trades.exit_date > d.index[-1]))).sum()))
        win = float(closed.loc[closed.net_pnl > 0, "net_pnl"].sum())
        loss = float(-closed.loc[closed.net_pnl < 0, "net_pnl"].sum())
        metrics["complete_trade_profit_factor"] = win / loss if loss > 0 else np.nan
    return metrics


def attribute_trades(result: StrategyResult, ev: Evaluation, train_end: pd.Timestamp) -> pd.DataFrame:
    trades = result.trades.copy()
    if trades.empty:
        return trades.assign(gross_pnl=[], costs=[], net_pnl=[])
    gross_ids = result.trade_ids_at_close.shift(1).fillna(0).to_numpy(dtype=int)
    fee_ids = result.cost_trade_ids_at_close.shift(1).fillna(0).to_numpy(dtype=int)
    gross = ev.gross_by_instrument.fillna(0).to_numpy()
    costs = ev.costs_by_instrument.fillna(0).to_numpy()
    max_id = int(trades.trade_id.max())
    for label, row_mask in [("", np.ones(len(ev.daily), dtype=bool)),
                            ("train_", ev.daily.index <= train_end),
                            ("forward_", ev.daily.index > train_end)]:
        g = np.bincount(gross_ids[row_mask].ravel(), weights=gross[row_mask].ravel(), minlength=max_id + 1)
        c = np.bincount(fee_ids[row_mask].ravel(), weights=costs[row_mask].ravel(), minlength=max_id + 1)
        trades[label + "gross_pnl"] = trades.trade_id.map(dict(enumerate(g)))
        trades[label + "costs"] = trades.trade_id.map(dict(enumerate(c)))
        trades[label + "net_pnl"] = trades[label + "gross_pnl"] - trades[label + "costs"]
    trades["initial_stop_risk_units"] = (trades.initial_quantity.abs()
                                          * (trades.entry_price - trades.initial_stop).abs())
    trades["pnl_per_initial_risk"] = trades.net_pnl / trades.initial_stop_risk_units.replace(0, np.nan)
    trades["entry_date"] = pd.to_datetime(trades.entry_date)
    trades["exit_date"] = pd.to_datetime(trades.exit_date)
    return trades


def instrument_metrics(ev: Evaluation, p: Parameters, mask: np.ndarray,
                       trades: pd.DataFrame) -> pd.DataFrame:
    rows = []
    dates = ev.daily.index[mask]
    for col in ev.net_by_instrument:
        pnl = ev.net_by_instrument.loc[mask, col]
        path = np.r_[0.0, pnl.fillna(0).cumsum().to_numpy()]
        entered = trades[(trades.instrument == col) & (trades.entry_date >= dates[0])
                         & (trades.entry_date <= dates[-1])] if not trades.empty else trades
        rows.append(dict(instrument=col, net_pnl=float(pnl.sum()),
                         gross_pnl=float(ev.gross_by_instrument.loc[mask, col].sum()),
                         costs=float(ev.costs_by_instrument.loc[mask, col].sum()),
                         net_sharpe=sharpe(pnl, p.annualization),
                         adjusted_sharpe=adjusted_sharpe(pnl, p.annualization),
                         max_drawdown_units=float((path - np.maximum.accumulate(path)).min()),
                         invested_fraction=float((ev.earning_positions.loc[mask, col].fillna(0).abs() > 0).mean()),
                         trades_entered=len(entered)))
    return pd.DataFrame(rows)


def bootstrap_sharpe(pnl: pd.Series, *, block: int = 20, samples: int = 1000,
                     seed: int = 42, annualization: int = 260) -> dict[str, Any]:
    values = pnl.dropna().to_numpy()
    rng = np.random.default_rng(seed)
    stats, means = [], []
    for _ in range(samples):
        starts = rng.integers(0, len(values), size=int(np.ceil(len(values) / block)))
        indices = ((starts[:, None] + np.arange(block)) % len(values)).ravel()[:len(values)]
        sample = values[indices]
        std = sample.std(ddof=1)
        if std > 1e-15:
            stats.append(sample.mean() / std * np.sqrt(annualization))
        means.append(sample.mean() * annualization)
    return dict(block_days=block, resamples=samples, seed=seed,
                sharpe_ci95_low=float(np.quantile(stats, .025)) if stats else np.nan,
                sharpe_ci95_high=float(np.quantile(stats, .975)) if stats else np.nan,
                annual_mean_pnl_ci95_low=float(np.quantile(means, .025)),
                annual_mean_pnl_ci95_high=float(np.quantile(means, .975)))

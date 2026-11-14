"""Readable, causal Lynx momentum engine. Also embedded in the research notebook."""
from dataclasses import dataclass, asdict, replace
from functools import lru_cache
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf
from joblib import Parallel, delayed


@dataclass(frozen=True)
class Config:
    signal: str = 'xs_risk'
    lookback: int = 126
    skip: int = 21
    frequency: int = 5
    rank_vol: int = 63
    size_vol: int = 63
    fraction: float = 0.30
    weighting: str = 'simple'
    covariance: int = 126
    penalty: float = 0.05
    overlay: str = 'none'
    risk_window: int = 63
    threshold_history: int = 252
    threshold_quantile: float = 0.90
    partial: float = 1.0
    band: float = 0.0
    warmup: int = 1008


def read_prices(path):
    """Keep the supplied observation calendar; never resample, fill, or backfill."""
    frame = pd.read_csv(path, index_col='AsOfDate', parse_dates=True)
    close = frame.loc[:, frame.columns.str.endswith('|Close')].copy()
    close.columns = close.columns.str.removesuffix('|Close')
    assert len(close.columns) and close.columns.is_unique
    assert close.index.is_unique and close.index.is_monotonic_increasing
    if close.isna().any().any() or not np.isfinite(close.to_numpy()).all():
        raise ValueError('Close data contain gaps: specify a causal missing-quote policy first.')
    return close


def score(pnl, pearson=False):
    """PDF equation (5), with annualized S INSIDE the adjustment. No clipping."""
    y = np.asarray(pnl, dtype=float)
    y = y[np.isfinite(y)]
    n = len(y)
    mean = y.mean() if n else np.nan
    sd = y.std(ddof=1) if n > 1 else np.nan
    empty = dict(adjusted=np.nan, sharpe=np.nan, mean=mean, sd=sd,
                 skew=np.nan, excess=np.nan, factor=np.nan)
    if n < 4 or not np.isfinite(sd) or sd <= 1e-14:
        return empty
    s = np.sqrt(252) * mean / sd
    sk = stats.skew(y, bias=False)
    ex = stats.kurtosis(y, fisher=True, bias=False)
    factor = 1 + s * sk / 6 - s**2 * (ex + (3 if pearson else 0)) / 24
    return dict(adjusted=s*factor, sharpe=s, mean=mean, sd=sd,
                skew=sk, excess=ex, factor=factor)


def backtest(prices, positions, cost=0.02):
    """Decide t, execute close t+1, book P&L AND the specified fee at t+2.

    Positions start at zero. A reversal +a -> -a trades 2a units.
    The final two desired rows have not yet generated P&L/fees in this sample.
    There is no artificial terminal liquidation.
    """
    x = positions.reindex_like(prices).fillna(0.0)
    delta = x.diff()
    delta.iloc[0] = x.iloc[0]  # entry relative to the zero position before the file
    held = x.shift(2).fillna(0.0)
    units = delta.shift(2).fillna(0.0).abs()
    asset_gross = held * prices.diff().fillna(0.0)
    asset_fees = cost * units
    asset_net = asset_gross - asset_fees
    daily = pd.DataFrame({'gross':asset_gross.sum(axis=1),
                          'fees':asset_fees.sum(axis=1),
                          'net':asset_net.sum(axis=1),
                          'traded_units':units.sum(axis=1),
                          'held_units':held.abs().sum(axis=1)}, index=prices.index)
    return dict(x=x, held=held, delta=delta, asset_gross=asset_gross,
                asset_fees=asset_fees, asset_net=asset_net, daily=daily)


class MomentumEngine:
    """One shared implementation; cached trailing inputs, independent configurations."""
    def __init__(self, prices):
        self.p = prices
        self.dp = prices.diff()
        self.values = prices.to_numpy()
        self.changes = self.dp.to_numpy()

    @lru_cache(maxsize=None)
    def volatility(self, window):
        # Sample standard deviation of price CHANGES, not percentage returns.
        return self.dp.rolling(window, min_periods=window).std(ddof=1).to_numpy()

    @lru_cache(maxsize=None)
    def signal(self, kind, lookback, skip, rank_vol):
        change = self.p.shift(skip) - self.p.shift(skip + lookback)
        if kind in ('xs_risk', 'ts'):
            denom = pd.DataFrame(self.volatility(rank_vol), index=self.p.index,
                                 columns=self.p.columns).shift(skip) * np.sqrt(lookback)
            return (change / denom.where(denom > 1e-10)).to_numpy()
        if kind == 'xs_return':
            # Relative-price momentum is a matched alternative, not external returns.
            return (self.p.shift(skip) / self.p.shift(skip+lookback) - 1).to_numpy()
        raise ValueError(kind)

    def _allocation(self, t, c, sig, vol):
        n = len(vol)
        valid = np.isfinite(sig) & np.isfinite(vol) & (vol > 1e-10)
        ids = np.flatnonzero(valid)
        w = np.zeros(n)
        if len(ids) < 6:
            return w, None
        ordered = ids[np.argsort(sig[ids], kind='stable')]
        if c.signal == 'ts':
            longs = ids[sig[ids] > 0]
            shorts = ids[sig[ids] < 0]
        else:
            k = max(1, int(len(ids)*c.fraction))
            shorts, longs = ordered[:k], ordered[-k:]
        selected = np.r_[longs, shorts]
        signs = np.r_[np.ones(len(longs)), -np.ones(len(shorts))]
        if not len(selected):
            return w, None
        a0 = np.r_[np.full(len(longs), 0.5/max(1,len(longs))),
                   np.full(len(shorts), 0.5/max(1,len(shorts)))]
        a = a0.copy()
        fit = None
        if c.weighting == 'tilt':
            for side in [np.arange(len(longs)), np.arange(len(longs),len(selected))]:
                if len(side):
                    strength = np.abs(sig[selected[side]])
                    order = stats.rankdata(strength, method='average')
                    tilts = 0.5 + order / len(side)
                    a[side] = 0.5 * tilts/tilts.sum()
        elif c.weighting == 'lw':
            # Every selected name's entire history is standardized by TODAY'S sigma.
            history = self.changes[t-c.covariance+1:t+1, selected] / vol[selected]
            if history.shape[0] != c.covariance or not np.isfinite(history).all():
                return w, None
            covariance = LedoitWolf().fit(history).covariance_
            signed_cov = covariance * np.outer(signs, signs)
            side_rows = []
            if len(longs):
                side_rows.append(np.r_[np.ones(len(longs)), np.zeros(len(shorts))])
            if len(shorts):
                side_rows.append(np.r_[np.zeros(len(longs)), np.ones(len(shorts))])
            A = np.asarray(side_rows)
            objective = lambda z: z@signed_cov@z + c.penalty*np.sum((z-a0)**2)
            gradient = lambda z: 2*signed_cov@z + 2*c.penalty*(z-a0)
            result = minimize(objective, a0, jac=gradient, method='SLSQP',
                              bounds=list(zip(0.5*a0,1.5*a0)),
                              constraints={'type':'eq','fun':lambda z:A@z-0.5,
                                           'jac':lambda z:A},
                              options={'ftol':1e-10, 'maxiter':100})
            ok = bool(result.success and np.max(np.abs(A@result.x-0.5)) < 1e-7)
            a = result.x if ok else a0
            # Audit is attached to the raw target, before exposure scaling/smoothing.
            fit = dict(date=self.p.index[t], success=ok,
                       minimum_ratio=float(np.min(a/a0)),
                       maximum_ratio=float(np.max(a/a0)),
                       long_sum=float(a[:len(longs)].sum()),
                       short_sum=float(a[len(longs):].sum()),
                       covariance_min_eigenvalue=float(np.linalg.eigvalsh(covariance)[0]))
        w[selected] = signs*a
        return w / vol, fit

    def positions(self, c):
        """No training labels: trailing covariance and risk fits end at decision t."""
        sig = self.signal(c.signal,c.lookback,c.skip,c.rank_vol)
        vols = self.volatility(c.size_vol)
        nrows, nassets = self.values.shape
        base = np.zeros((nrows,nassets))
        audit = []
        previous = np.zeros(nassets)
        for t in range(nrows):
            # Anchor schedule to the first supplied row, independent of file end.
            if t >= c.warmup and t % c.frequency == 0:
                previous, fit = self._allocation(t,c,sig[t],vols[t])
                if fit is not None:
                    audit.append(fit)
            base[t] = previous
        base_frame = pd.DataFrame(base,index=self.p.index,columns=self.p.columns)
        reference_gross = (base_frame.shift(2).fillna(0)*self.dp.fillna(0)).sum(axis=1)
        risk = reference_gross.rolling(c.risk_window,min_periods=c.risk_window).std(ddof=1)
        # Prior risk estimates set the threshold/reference; current risk includes t.
        past = risk.where(risk > 1e-10).shift(1).rolling(
            c.threshold_history,min_periods=c.threshold_history)
        threshold = past.quantile(c.threshold_quantile)
        target_risk = past.median()
        if c.overlay == 'continuous':
            multiplier = (target_risk/risk).clip(0.25,1.5).fillna(0.0)
        elif c.overlay == 'conditional':
            multiplier = pd.Series(1.0,index=self.p.index)
            high = risk > threshold
            multiplier.loc[high] = (threshold[high]/risk[high]).clip(0.25,1.0)
            multiplier.loc[threshold.isna()] = 0.0
        else:
            multiplier = pd.Series(1.0,index=self.p.index)
        x = np.zeros_like(base)
        pre_smoothing = np.zeros_like(base)
        applied_multiplier = np.zeros(nrows)
        previous = np.zeros(nassets)
        target = np.zeros(nassets)
        applied = 0.0
        for t in range(nrows):
            if t >= c.warmup and t % c.frequency == 0:
                applied = multiplier.iloc[t]
                target = base[t]*applied
                trade = target-previous
                # Portfolio-wide no-trade band in CURRENT risk units; then partial trade.
                risk_distance = np.nansum(np.abs(trade)*vols[t])
                if risk_distance >= c.band:
                    previous = previous + c.partial*trade
            x[t] = previous
            pre_smoothing[t] = target
            applied_multiplier[t] = applied
        frame = pd.DataFrame(x,index=self.p.index,columns=self.p.columns)
        diagnostics = pd.DataFrame({'reference_vol':risk, 'threshold':threshold,
                                    'reference_target':target_risk,
                                    'requested_multiplier':multiplier,
                                    'applied_multiplier':applied_multiplier},index=self.p.index)
        return dict(x=frame, base=base_frame,
                    target=pd.DataFrame(pre_smoothing,index=self.p.index,columns=self.p.columns),
                    diagnostic=diagnostics, audit=pd.DataFrame(audit))


def evaluate(engine, c, cost=0.02):
    generated = engine.positions(c)
    return {'config':c, **generated, **backtest(engine.p,generated['x'],cost)}


def evaluate_many(engine, configs, cost=0.02, n_jobs=4, compact=True):
    # Thread workers share cached arrays. Pre-warm caches before concurrent reads.
    for c in configs:
        engine.volatility(c.size_vol)
        engine.signal(c.signal,c.lookback,c.skip,c.rank_vol)
    def run(c):
        result = evaluate(engine,c,cost)
        return {k:result[k] for k in ['config','daily']} if compact else result
    return Parallel(n_jobs=n_jobs, prefer='threads')(delayed(run)(c) for c in configs)


def combine(engine, configs, cost=0.02):
    """Equal sleeves; net their trades BEFORE charging costs (not average sleeve fees)."""
    sleeves = [engine.positions(c)['x'] for c in configs]
    x = sum(sleeves)/len(sleeves)
    return dict(config=tuple(configs), **backtest(engine.p,x,cost))


def drawdown(pnl):
    curve = pnl.cumsum()
    return curve - curve.cummax().clip(lower=0)


def scorecard(result, dates):
    d = result['daily'].loc[dates]
    s = score(d.net)
    traded = d.traded_units.sum()
    return dict(adjusted=s['adjusted'], net_sharpe=s['sharpe'],
                gross_sharpe=score(d.gross)['sharpe'], mean_net=s['mean'],
                skew=s['skew'], excess=s['excess'], factor=s['factor'],
                pearson_adjusted=score(d.net,pearson=True)['adjusted'],
                holding_days=2*d.held_units.sum()/traded if traded > 0 else np.nan,
                turnover=traded/(2*d.held_units.sum()) if d.held_units.sum()>0 else 0,
                fees=d.fees.sum(), net_pnl=d.net.sum(), gross_pnl=d.gross.sum(),
                max_drawdown=-drawdown(d.net).min(), traded_units=traded,
                break_even_cost=d.gross.sum()/traded if traded>0 else np.nan,
                observations=len(d))


def config_to_dict(c):
    return {'sleeves':[asdict(x) for x in c]} if isinstance(c,tuple) else asdict(c)


def config_from_dict(d):
    return tuple(Config(**x) for x in d['sleeves']) if 'sleeves' in d else Config(**d)


def generate_daily_targets(price_file, configuration, output_file=None):
    """Replay the full available history (same origin) for a new organizer-format CSV.

    Return desired X[t], not lagged realized positions. Append new rows to the original
    history; replay preserves rolling fits, rebalance phase and partial-trade state.
    """
    flat = False
    if isinstance(configuration,(str,Path)):
        wrapper = json.loads(Path(configuration).read_text())
        flat = wrapper.get('submission_is_flat',False)
        configuration = wrapper['submission_config']
    c = config_from_dict(configuration) if isinstance(configuration,dict) else configuration
    p = read_prices(price_file)
    engine = MomentumEngine(p)
    x = combine(engine,c)['x'] if isinstance(c,tuple) else engine.positions(c)['x']
    if flat:
        x = x*0
    x.index.name = 'AsOfDate'
    if output_file:
        x.to_csv(output_file)
    return x

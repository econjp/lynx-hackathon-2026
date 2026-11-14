"""Cost-conscious trend extension; all inputs are available at the decision close."""
from dataclasses import dataclass, asdict, replace
import numpy as np
import pandas as pd
from lynx_engine import read_prices, backtest, scorecard, MomentumEngine


@dataclass(frozen=True)
class SlowConfig:
    lookbacks: tuple = (252,)
    entry: float = 1.0
    exit: float = 0.25
    frequency: int = 21
    rank_vol: int = 63
    size_vol: int = 63
    resize_band: float = 0.50
    agreement: bool = False
    cost_gate: float = 0.0  # zero disables; maximum fee / daily price-change volatility
    warmup: int = 1008
    total_risk_budget: float = 1.0


def slow_positions(prices, config, cost=0.02):
    """Fixed risk allotment per name: new entries never resize unrelated positions.

    Unused allotments stay in cash. The portfolio is not forced to have a 50/50 side
    split. This intentional change is isolated by the matched N00/N01 controls.
    Entry/exit and volatility resizing are checked on the same fixed row schedule.
    """
    c = config
    assert c.entry >= c.exit >= 0
    vol = prices.diff().rolling(c.size_vol).std(ddof=1).to_numpy()
    rank_vol = prices.diff().rolling(c.rank_vol).std(ddof=1)
    component = np.stack([((prices-prices.shift(L))/(rank_vol*np.sqrt(L))).to_numpy()
                          for L in c.lookbacks])
    signal = np.mean(component, axis=0)
    count, names = prices.shape
    allotment = c.total_risk_budget/names
    current = np.zeros(names)
    anchor_vol = np.zeros(names)
    x = np.zeros((count,names))
    events = []
    anchors = np.zeros_like(x)
    for t in range(count):
        if t >= c.warmup and t % c.frequency == 0:
            for j in range(names):
                sigma, s = vol[t,j], signal[t,j]
                if not np.isfinite(sigma+s) or sigma <= 1e-10:
                    continue
                side = np.sign(current[j])
                desired_side = np.sign(s)
                agreement = (not c.agreement or
                             np.all(np.sign(component[:,t,j]) == desired_side))
                cheap = c.cost_gate <= 0 or cost/sigma <= c.cost_gate
                strong = abs(s) > c.entry and agreement and cheap
                target_side = side
                if side == 0:
                    if strong:
                        target_side = desired_side
                elif strong and desired_side != side:
                    target_side = desired_side
                elif side*s <= c.exit:
                    target_side = 0
                reason = None
                if target_side != side:
                    reason = 'entry' if side == 0 else ('exit' if target_side == 0 else 'reversal')
                    current[j] = target_side*allotment/sigma
                    anchor_vol[j] = sigma if target_side else 0
                elif side != 0:
                    # Symmetric ratio band around the volatility used for last sizing.
                    ratio = sigma/anchor_vol[j]
                    if ratio > 1+c.resize_band or ratio < 1/(1+c.resize_band):
                        reason = 'risk_resize'
                        current[j] = side*allotment/sigma
                        anchor_vol[j] = sigma
                if reason:
                    events.append((prices.index[t],prices.columns[j],reason))
        x[t] = current
        anchors[t] = anchor_vol
    return {'x':pd.DataFrame(x,index=prices.index,columns=prices.columns),
            'anchor_vol':pd.DataFrame(anchors,index=prices.index,columns=prices.columns),
            'signal':pd.DataFrame(signal,index=prices.index,columns=prices.columns),
            'vol':pd.DataFrame(vol,index=prices.index,columns=prices.columns),
            'events':pd.DataFrame(events,columns=['date','asset','reason'])}


def slow_evaluate(prices, config, cost=0.02):
    generated = slow_positions(prices,config,cost)
    return {'config':config, **generated, **backtest(prices,generated['x'],cost)}


def generate_slow_targets(price_file, config, output_file=None, flat=False):
    if isinstance(config,dict):
        config = SlowConfig(**{**config,'lookbacks':tuple(config['lookbacks'])})
    x = slow_positions(read_prices(price_file),config)['x']
    if flat:
        x = x*0
    if output_file:
        x.to_csv(output_file,index_label='AsOfDate')
    return x

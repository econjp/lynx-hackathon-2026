"""Generate the slow-trend extension's desired daily positions from a full price history."""
import argparse,json
from pathlib import Path
from slow_trend_engine import generate_slow_targets
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('price_file');p.add_argument('--output',default='slow_targets.csv')
p.add_argument('--config',default='research_output/slow_trend/strategy_config.json')
p.add_argument('--fallback',action='store_true')
a=p.parse_args();cfg=json.loads(Path(a.config).read_text())
generate_slow_targets(a.price_file,cfg['fallback_config' if a.fallback else 'primary_config'],
                      a.output,flat=cfg['submission_is_flat'] and not a.fallback)
print('Saved',a.output)

"""Replay daily desired targets with the saved research configuration."""
import argparse
import json
from pathlib import Path
from lynx_engine import generate_daily_targets

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('price_file')
parser.add_argument('--config',default='research_output/strategy_config.json')
parser.add_argument('--output',default='daily_target_positions.csv')
parser.add_argument('--fallback',action='store_true')
args=parser.parse_args()
settings=json.loads(Path(args.config).read_text())
c=settings['fallback_config'] if args.fallback else settings['submission_config']
x=generate_daily_targets(args.price_file,c)
if settings.get('submission_is_flat',False) and not args.fallback:
    x=x*0
x.to_csv(args.output,index_label='AsOfDate')
print(f'Wrote {len(x)} rows and {len(x.columns)} assets to {args.output}')

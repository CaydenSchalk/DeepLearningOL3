import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def run_curve(csv):
    results = pd.read_csv(csv)
    rows = results[results['val_acc'].notna()]
    loss = rows['val_loss']
    n_nan = int(loss.isna().sum())
    return loss.interpolate(limit_direction='both').to_numpy(), n_nan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, default=HERE)
    parser.add_argument('--overwrite', action='store_true',
                        help='replace existing loss-results files')
    args = parser.parse_args()

    for exp_dir in sorted((args.output_dir / 'runs').iterdir()):
        csvs = sorted((exp_dir / 'logs').glob('run-*/version_*/metrics.csv'))
        if not csvs:
            continue
        target = args.output_dir / f'{exp_dir.name}-loss-results.txt'
        curves, nans = zip(*(run_curve(c) for c in csvs))
        info = f'{exp_dir.name}: {len(curves)} runs, epochs {[len(c) for c in curves]}, interpolated {list(nans)}'
        if target.exists() and not args.overwrite:
            print(info, '-> skipped, exists:', target.name)
            continue
        if len({len(c) for c in curves}) != 1:
            print(info, '-> skipped, runs differ in length')
            continue
        np.savetxt(target, np.array(curves), fmt='%.8f')
        print(info, '-> wrote', target.name)


if __name__ == '__main__':
    main()

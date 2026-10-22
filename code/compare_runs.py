"""
Summarize the per-case test results of several runs (CSV files from evaluate_pancreas_lcc.py --csv)
as markdown tables, with paired Wilcoxon signed-rank tests over the test scans.

Files are named <model>_<setting>_seed<seed>.csv, e.g. unet_3D_dycon_seed1337.csv, as written by
scripts/evaluate_all.sh and scripts/kaggle_train.sh. When a configuration has several seeds, each
scan's metric is first averaged over the seeds, and the spread of the mean Dice across seeds is
reported in a separate table.

Run from the code/ directory:
    python compare_runs.py --results ../results/v2
"""
import os
import re
import csv
import glob
import argparse
import warnings
from collections import defaultdict

import numpy as np
from scipy.stats import wilcoxon

MODELS = {'unet_3D': '3D U-Net', 'swin_umamba': 'Swin-UMamba'}
SETTINGS = {'sup12': '12 labeled only', 'dycon': 'DyCON, 12 labeled', 'dycon40k': 'DyCON, 12 labeled, 40k iterations',
            'full': 'Fully supervised'}
METRICS = [('dice', 'Dice (%) ↑'), ('hd95', 'HD95 ↓'), ('asd', 'ASD ↓')]
FILE_NAME = re.compile(r'^(unet_3D|swin_umamba)_(sup12|dycon|dycon40k|full)_seed(\d+)\.csv$')


def load_runs(results_dir):
    """{(model, setting): {seed: {case: {column: value}}}} from the CSV files in `results_dir`."""
    runs = defaultdict(dict)
    for path in sorted(glob.glob(os.path.join(results_dir, '*.csv'))):
        match = FILE_NAME.match(os.path.basename(path))
        if not match:
            continue
        model, setting, seed = match.groups()
        with open(path, newline='') as f:
            runs[(model, setting)][int(seed)] = {row['case']: {k: float(v) for k, v in row.items() if k != 'case'}
                                                 for row in csv.DictReader(f)}
    return runs


def per_case(runs, key, column, cases):
    """Per-case values of `column` for one configuration, averaged over its seeds."""
    seeds = runs[key]
    return np.array([[seeds[s][c][column] for c in cases] for s in sorted(seeds)]).mean(axis=0)


def p_value(a, b):
    """Two-sided paired Wilcoxon signed-rank p-value (NaN when every difference is zero)."""
    if np.allclose(a, b):
        return float('nan')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return float(wilcoxon(a, b).pvalue)


def fmt_p(p):
    return '–' if np.isnan(p) else ('< 0.001' if p < 0.001 else f'{p:.3f}')


def main():
    parser = argparse.ArgumentParser(description="Tables and paired tests from per-case result CSVs.")
    parser.add_argument('--results', type=str, default='../results/v2', help='Folder with <model>_<setting>_seed<seed>.csv')
    parser.add_argument('--out', type=str, default=None, help='Also write the markdown to this file')
    args = parser.parse_args()

    runs = load_runs(args.results)
    if not runs:
        raise SystemExit(f"No result files found in {args.results}")
    configs = [(m, s) for m in MODELS for s in SETTINGS if (m, s) in runs]
    cases = sorted(set.intersection(*(set(r) for seeds in runs.values() for r in seeds.values())))
    lines = [f"Test scans: {len(cases)}. Values are mean ± std over the scans. With several seeds, each scan is "
             "first averaged over the seeds.", ""]

    for metric, title in METRICS:
        lines += [f"**{title}**", "",
                  "| Setting | " + " | ".join(f"{MODELS[m]} raw | {MODELS[m]} + LCC" for m in MODELS) + " |",
                  "|---|" + ":---:|" * (2 * len(MODELS))]
        for setting, setting_name in SETTINGS.items():
            cells = []
            for model in MODELS:
                for stage in ('raw', 'lcc'):
                    if (model, setting) in runs:
                        v = per_case(runs, (model, setting), f'{stage}_{metric}', cases)
                        cells.append(f"{v.mean():.2f} ± {v.std():.2f}")
                    else:
                        cells.append('–')
            lines.append(f"| {setting_name} | " + " | ".join(cells) + " |")
        lines.append("")

    lines += ["**Mean Dice (%) across seeds**", "", "| Configuration | Seeds | Raw | + LCC |", "|---|---|:---:|:---:|"]
    for model, setting in configs:
        seeds = runs[(model, setting)]
        means = {stage: [np.mean([seeds[s][c][f'{stage}_dice'] for c in cases]) for s in sorted(seeds)]
                 for stage in ('raw', 'lcc')}
        cells = [f"{np.mean(v):.2f} ± {np.std(v):.2f}" if len(v) > 1 else f"{v[0]:.2f}" for v in means.values()]
        lines.append(f"| {MODELS[model]}, {SETTINGS[setting]} | {', '.join(map(str, sorted(seeds)))} | " + " | ".join(cells) + " |")
    lines.append("")

    comparisons = []
    for key in configs:
        for metric, title in METRICS:
            comparisons.append((f"{MODELS[key[0]]}, {SETTINGS[key[1]]}: + LCC vs. raw", title, key, f'lcc_{metric}', key, f'raw_{metric}'))
    for model in MODELS:
        for a, b in (('dycon', 'sup12'), ('dycon40k', 'sup12'), ('dycon40k', 'dycon'), ('full', 'dycon')):
            for stage in ('raw', 'lcc'):
                comparisons.append((f"{MODELS[model]}: {SETTINGS[a]} vs. {SETTINGS[b]} ({stage})", 'Dice (%) ↑',
                                    (model, a), f'{stage}_dice', (model, b), f'{stage}_dice'))
    for setting in SETTINGS:
        for stage in ('raw', 'lcc'):
            comparisons.append((f"{SETTINGS[setting]}: 3D U-Net vs. Swin-UMamba ({stage})", 'Dice (%) ↑',
                                ('unet_3D', setting), f'{stage}_dice', ('swin_umamba', setting), f'{stage}_dice'))

    lines += ["**Paired Wilcoxon signed-rank tests (two-sided, over the test scans)**", "",
              "| Comparison | Metric | Mean difference | p |", "|---|---|:---:|:---:|"]
    for name, title, key_a, col_a, key_b, col_b in comparisons:
        if key_a not in runs or key_b not in runs:
            continue
        a, b = per_case(runs, key_a, col_a, cases), per_case(runs, key_b, col_b, cases)
        lines.append(f"| {name} | {title} | {np.mean(a - b):+.2f} | {fmt_p(p_value(a, b))} |")

    text = "\n".join(lines) + "\n"
    print(text)
    if args.out:
        with open(args.out, 'w') as f:
            f.write(text)


if __name__ == '__main__':
    main()

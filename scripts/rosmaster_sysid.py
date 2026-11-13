"""Identify ROSMASTER base/arm response from REAL recorded episodes -> candidate file (never applied automatically).

  .venv-sim/bin/python scripts/rosmaster_sysid.py rosmaster_dock         # real catalog episodes
  .venv-sim/bin/python scripts/rosmaster_sysid.py rosmaster_pick_place --fixture   # self-check on synthetic recordings
Writes DATA/sysid/rosmaster-<profile>-<stamp>.json and prints held-out errors: nominal sim vs simulator form vs full model.
"""
import argparse, json, sys, tempfile, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from explorer3.sim import rosmaster_sysid as sysid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('profile', choices=sorted(sysid.CHANNELS))
    parser.add_argument('--fixture', action='store_true', help='synthetic self-check, nothing from the catalog')
    args = parser.parse_args()
    if args.fixture:
        params = ([{'delay_s': .08, 'tau_s': .15, 'gain': .8}] * 3 if args.profile == 'rosmaster_dock'
                  else [{'delay_s': .06, 'tau_s': .1, 'vmax_deg_s': 90.}] * 5)
        tmp = Path(tempfile.mkdtemp(prefix='rosmaster-sysid-fixture-'))
        entries = [{'id': f'fixture{i}', 'dataset': str(sysid.synthetic_episode(tmp / str(i), args.profile, params, i))} for i in range(4)]
        report = sysid.identify(entries, args.profile, allow_synthetic=True)
        report['fixture_truth'] = params
        out = tmp / 'candidate.json'
    else:
        from explorer3 import catalog
        from explorer3.paths import DATA
        entries = sorted((e for e in catalog.rows('episodes') if e['profile'] == args.profile), key=lambda e: e.get('created') or 0)
        report = sysid.identify(entries, args.profile)
        report['simulator_values'] = sysid.simulator_values(report)
        out = DATA / 'sysid' / f'rosmaster-{args.profile}-{time.strftime("%Y%m%d-%H%M%S")}.json'
        out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(out)
    for name, channel in report['channels'].items():
        if 'identified' not in channel:
            print(f'{name:9s} {channel["status"]}'); continue
        i, h = channel['identified'], channel['holdout_rmse']
        print(f'{name:9s} delay {i["delay_s"]:.2f}s tau {i["tau_s"]:.3f}s gain {i["gain"]:.3f} vmax {i.get("vmax_deg_s")} | '
              f'held-out RMSE nominal {h["nominal_sim"]:.4f} sim-form {h["simulator_form"]:.4f} full {h["identified"]:.4f} {channel["units"]}')


if __name__ == '__main__':
    main()

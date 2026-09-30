import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = ROOT / "code"

PIPELINE = [
    "01_publication_analysis.py",
    "02_factor_proxy_definition_horse_race.py",
    "03_plot_weekly_knn_cost_panels.py",
    "04_plot_weekly_logit_winners_losers_cost_panels.py",
    "05_plot_winners_losers_k_sharpe.py",
    "06_make_payoff_asymmetry_table.py",
]


def main() -> None:
    for script in PIPELINE:
        script_path = CODE_DIR / script
        print(f"\n=== Running {script} ===", flush=True)
        subprocess.run([sys.executable, str(script_path)], check=True, cwd=str(ROOT))
    print("\nReplication pipeline complete. Outputs are in the outputs/ folder.")


if __name__ == "__main__":
    main()

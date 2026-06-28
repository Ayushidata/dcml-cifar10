"""
==============================================================
RUN ALL EXPERIMENTS — MASTER SCRIPT
==============================================================
Runs Experiment A → B → C in sequence.
All results saved to ./results/
Usage: python run_all_experiments.py
==============================================================
"""

import subprocess, sys, os

def run(script):
    print(f"\n{'='*60}")
    print(f"  RUNNING: {script}")
    print(f"{'='*60}\n")
    result = subprocess.run([sys.executable, script], check=True)
    return result.returncode

if __name__ == "__main__":
    os.makedirs("./results", exist_ok=True)
    for script in [
        "experiment_A_fixed.py",
        "experiment_B_fixed.py",
        "experiment_C_fixed.py",
    ]:
        run(script)
    print("\n" + "="*60)
    print("  ALL EXPERIMENTS COMPLETE!")
    print("  Results saved to ./results/")
    print("="*60)

"""
Retrains the demand forecasting model end-to-end:
  1. preprocessing  — aggregate raw parquet -> processed_data/aggregated.parquet
  2. features       — build lag/rolling features -> processed_data/features.parquet
  3. train          — fit LightGBM model -> models/model.lgb
"""

import sys
import time

import data.preprocessing as preprocessing
import data.features as features
import training.train as train


def step(name: str, fn) -> None:
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    t0 = time.time()
    fn()
    print(f"  Done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    total = time.time()
    try:
        step("Step 1/3 — Preprocessing", preprocessing.main)
        step("Step 2/3 — Feature engineering", features.main)
        step("Step 3/3 — Training", train.main)
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"\nRetrain complete in {time.time() - total:.1f}s")

#!/usr/bin/env python3
"""
Run real-data benchmark for Adaptive Fusion Engine.
Uses production-style company data to quantify volatility reduction.
No UI, no production code changes.
"""
import sys

from services.evaluation import run_real_data_benchmark, print_benchmark_report


def main():
    try:
        results = run_real_data_benchmark(min_companies=10)
        if len(results) >= 10:
            print_benchmark_report(results)
        else:
            print(
                f"Processed {len(results)} companies (need 10+ with sufficient daily data)."
            )
            if results:
                print_benchmark_report(results)
    except Exception as e:
        print(f"Benchmark failed: {e}", file=sys.stderr)
        print(
            "Ensure Data/dec30_to_jan12/ exists with data_as_csv.csv",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()

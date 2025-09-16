#!/usr/bin/env python3
"""
Inspect the slower-decay weighting curve used for lane probability aggregation.

Weights are defined as: w_t = -1/6 * ln(t+1) + 1, for t in [0, T_future-1].
This script prints stats, renders an ASCII plot, and tries to save a PNG plot
if matplotlib is available.
"""

import argparse
from typing import List


def compute_weights(T: int) -> List[float]:
    import math
    # w_t = -1/6 * ln(t+1) + 1
    return [max(0.0, 1.0 - (math.log(t + 1) / 6.0)) for t in range(T)]


def normalize(weights: List[float]) -> List[float]:
    s = sum(weights)
    return [w / s for w in weights] if s > 0 else weights


def ascii_plot(weights: List[float], width: int = 50, title: str = "weights") -> None:
    max_w = max(weights) if weights else 1.0
    print(f"\nASCII plot: {title}")
    for t, w in enumerate(weights):
        bar_len = int((w / max_w) * width)
        print(f"t={t:3d} | {'█' * bar_len}")


def try_save_png(weights: List[float], output: str) -> bool:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return False

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(len(weights)), weights, marker='o', ms=3)
    ax.set_title('Inverse Weights Curve: w_t = 1/(t+1)')
    ax.set_xlabel('t (future step index)')
    ax.set_ylabel('weight')
    ax.grid(True, linestyle='--', alpha=0.4)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return True


def main():
    parser = argparse.ArgumentParser(description="Inspect inverse weights curve w_t = 1/(t+1)")
    parser.add_argument("--future-steps", "-T", type=int, default=80, help="Number of future steps (default: 80)")
    parser.add_argument("--no-ascii", action="store_true", help="Do not print ASCII plots")
    parser.add_argument("--save", type=str, default="weights_curve.png", help="Output PNG file if matplotlib is available")
    args = parser.parse_args()

    T = max(1, args.future_steps)
    weights = compute_weights(T)
    weights_norm = normalize(weights)

    print(f"T_future = {T}")
    print(f"First 10 raw weights:      {weights[:10]}")
    print(f"First 10 normalized:       {weights_norm[:10]}")
    print(f"Sum(raw) = {sum(weights):.6f}, Sum(norm) = {sum(weights_norm):.6f}")

    if not args.no_ascii:
        ascii_plot(weights, title="raw")
        ascii_plot(weights_norm, title="normalized")

    if try_save_png(weights, args.save):
        print(f"Saved PNG plot to: {args.save}")
    else:
        print("matplotlib not available; skipped PNG plot.")


if __name__ == "__main__":
    main()

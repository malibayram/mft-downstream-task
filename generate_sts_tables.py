#!/usr/bin/env python3
"""STS Benchmark Results Table Generator - Generates markdown tables and charts."""
import json
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np


def load_results(filename):
    """Load JSON results file."""
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)


def format_table(headers, rows):
    """Format data as a Markdown table."""
    if not rows:
        return "No data available."
    col_widths = [
        max(len(h), max(len(str(r[i])) for r in rows)) for i, h in enumerate(headers)
    ]
    fmt = "| " + " | ".join(f"{{:<{w}}}" for w in col_widths) + " |"
    sep = "| " + " | ".join("-" * w for w in col_widths) + " |"
    return "\n".join(
        [fmt.format(*headers), sep] + [fmt.format(*[str(c) for c in r]) for r in rows]
    )


def process_split_data(entries, split_name):
    """Group entries by model, assign Training Steps, find best results."""
    model_stats = {}
    for e in entries:
        model_stats.setdefault(e["model"], []).append(e)

    all_runs = []
    for results in model_stats.values():
        results.sort(key=lambda x: x["timestamp_obj"])
        for i, res in enumerate(results):
            step = {0: 0, 1: 50, 2: 100}.get(i, f"?({i})")
            if split_name == "train" and len(results) == 1:
                step = 100
            res["step"] = step
            all_runs.append(res)

    all_runs.sort(key=lambda x: x["timestamp_obj"])
    best_results = [
        max(results, key=lambda x: x.get("pearson", 0) or 0)
        for results in model_stats.values()
    ]
    best_results.sort(key=lambda x: x["timestamp_obj"], reverse=True)
    return all_runs, best_results


def generate_detailed_summary(entries_by_split):
    """Generate detailed textual summary with scores as percentages."""
    lines = ["# 📝 Detailed Analysis Summary\n"]
    lines.append(
        "> **Note:** The 'train' and 'test' labels refer to dataset split names in HuggingFace. Neither split was used during model training. This is a zero-shot evaluation.\n"
    )
    test_entries = entries_by_split.get("test", [])

    if not test_entries:
        return "\n".join(lines + ["*No test data available for analysis.*"])

    _, best_test_results = process_split_data(test_entries, "test")
    best_test_results.sort(key=lambda x: x.get("pearson", 0), reverse=True)
    best = best_test_results[0]

    lines.append("### 🏆 Overall Best Performance")
    lines.append(
        f"The best performing model on the **Test Split** is **{best['model']}** with a Pearson of **{best['pearson']*100:.2f}%** and Spearman of **{best['spearman']*100:.2f}%**.\n"
    )

    def avg_score(keyword):
        scores = [
            r["pearson"]
            for r in best_test_results
            if keyword in r["model"].lower() and "random" not in r["model"].lower()
        ]
        return sum(scores) / len(scores) if scores else 0.0

    avg_mft, avg_tabi = avg_score("mft"), avg_score("tabi")
    diff = avg_mft - avg_tabi
    lines.append("### ⚔️ MFT vs Tabi Tokenizer Comparison")
    lines.append(f"- **MFT Models Average**: {avg_mft*100:.2f}%")
    lines.append(f"- **Tabi Models Average**: {avg_tabi*100:.2f}%")
    lines.append(
        f"\nMFT models {'outperformed' if diff > 0 else 'underperformed'} Tabi by **{abs(diff)*100:.2f}** percentage points.\n"
    )

    avg_magibu, avg_gemma = avg_score("magibu"), avg_score("gemma")
    lines.append("### 🏗️ Base Model Comparison")
    lines.append(f"- **EmbeddingMagibu Average**: {avg_magibu*100:.2f}%")
    lines.append(f"- **EmbeddingGemma Average**: {avg_gemma*100:.2f}%")
    better = "EmbeddingMagibu" if avg_magibu > avg_gemma else "EmbeddingGemma"
    lines.append(f"\n**{better}** yields better results on average.\n")

    return "\n".join(lines)


def generate_bar_chart(model_data, output_filename, title):
    """Generate grouped vertical bar chart."""
    models = list(model_data.keys())
    pearson = [
        model_data[m][-1][1] * 100 if model_data[m] else 0 for m in models
    ]  # ×100
    spearman = [
        model_data[m][-1][2] * 100 if model_data[m] else 0 for m in models
    ]  # ×100

    x = np.arange(len(models))
    width = 0.35
    _, ax = plt.subplots(figsize=(10, 8))

    mft_idx = [i for i, m in enumerate(models) if "mft" in m.lower()]
    r1 = ax.bar(x - width / 2, pearson, width, label="Pearson", color="skyblue")
    r2 = ax.bar(x + width / 2, spearman, width, label="Spearman", color="orange")

    for i in mft_idx:
        for r in (r1[i], r2[i]):
            r.set_hatch("///")
            r.set_edgecolor("black")

    ax.set_ylabel("Score (%)")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.legend(loc="lower right")
    ax.grid(True, axis="y", linestyle="--", alpha=0.7)

    for i, lbl in enumerate(ax.get_xticklabels()):
        if "mft" in models[i].lower():
            lbl.set_fontweight("bold")
            lbl.set_color("#2c3e50")

    ax.bar_label(r1, padding=3, fmt="%.2f%%", rotation=90)
    ax.bar_label(r2, padding=3, fmt="%.2f%%", rotation=90)
    plt.margins(y=0.2)
    plt.tight_layout()
    plt.savefig(output_filename)
    plt.close()
    print(f"Bar Chart generated at {output_filename}")
    return True


def generate_line_chart(model_data, output_filename, title):
    """Generate line charts for Pearson and Spearman."""
    _, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

    for model, points in model_data.items():
        points.sort(key=lambda x: x[0])
        steps = [p[0] for p in points]
        ax1.plot(steps, [p[1] * 100 for p in points], marker="o", label=model)
        ax2.plot(steps, [p[2] * 100 for p in points], marker="o", label=model)

    for ax, metric in [(ax1, "Pearson"), (ax2, "Spearman")]:
        ax.set_title(f"{title} - {metric}")
        ax.set_xlabel("Training Step")
        ax.set_ylabel(f"{metric} Score (%)")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.grid(True)

    plt.tight_layout()
    plt.savefig(output_filename)
    plt.close()
    print(f"Line Chart generated at {output_filename}")
    return True


def generate_chart(all_runs, output_filename, title):
    """Generate chart - bar for single points, line for multiple."""
    model_data = {}
    for r in all_runs:
        if isinstance(r["step"], int):
            model_data.setdefault(r["model"], []).append(
                (r["step"], r["pearson"], r["spearman"])
            )

    if not model_data:
        return False

    max_pts = max(len(pts) for pts in model_data.values())
    return (
        generate_bar_chart(model_data, output_filename, title)
        if max_pts <= 1
        else generate_line_chart(model_data, output_filename, title)
    )


def main():
    filename = "sts_benchmark_results.json"
    if not os.path.exists(filename):
        print(f"File {filename} not found.")
        return

    try:
        data = load_results(filename)
    except json.JSONDecodeError as e:
        print(f"Error reading JSON: {e}")
        return

    # Parse entries
    all_entries = []
    for entry in data:
        ts_raw = entry.get("timestamp", "")
        ts_obj = datetime.min
        ts_display = ts_raw
        if ts_raw:
            try:
                ts_obj = datetime.fromisoformat(ts_raw)
                ts_display = ts_obj.strftime("%Y-%m-%d %H:%M:%S")
            except ValueError:
                pass

        for res in entry.get("results", []):
            all_entries.append(
                {
                    "timestamp_obj": ts_obj,
                    "timestamp_display": ts_display,
                    "dataset": entry.get("dataset", "unknown"),
                    "split": res.get("split", "unknown"),
                    "model": res.get("model", "Unknown"),
                    "pearson": res.get("pearson", 0.0) or 0.0,
                    "spearman": res.get("spearman", 0.0) or 0.0,
                    "num_samples": res.get("num_samples", 0),
                }
            )

    # Group by split
    entries_by_split = {}
    for e in all_entries:
        entries_by_split.setdefault(e["split"], []).append(e)

    output_lines = []

    # Summary at TOP
    output_lines.append(generate_detailed_summary(entries_by_split))

    # Process each split
    for split_name, entries in entries_by_split.items():
        processed_runs, best_results = process_split_data(entries, split_name)
        split_title = f"{split_name.capitalize()} Split"
        output_lines.append(f"# {split_title}\n")

        chart_file = f"sts_benchmark_chart_{split_name}.png"
        if generate_chart(
            processed_runs, chart_file, f"Model Performance - {split_title}"
        ):
            output_lines.append(f"![{split_title} Performance]({chart_file})\n")

        # All Runs Table (scores ×100)
        rows1 = []
        for r in processed_runs:
            name = f"**{r['model']}**" if "mft" in r["model"] else r["model"]
            p = (
                f"**{r['pearson']*100:.2f}%**"
                if "mft" in r["model"]
                else f"{r['pearson']*100:.2f}%"
            )
            s = (
                f"**{r['spearman']*100:.2f}%**"
                if "mft" in r["model"]
                else f"{r['spearman']*100:.2f}%"
            )
            rows1.append([r["timestamp_display"], name, r["step"], p, s])

        output_lines.append(f"## All Runs - {split_title}\n")
        output_lines.append(
            format_table(
                ["Timestamp", "Model", "Training Step", "Pearson", "Spearman"], rows1
            )
        )
        output_lines.append("\n")

        # Best Results Table (scores ×100)
        rows2 = []
        for r in best_results:
            name = f"**{r['model']}**" if "mft" in r["model"] else r["model"]
            p = (
                f"**{r['pearson']*100:.2f}%**"
                if "mft" in r["model"]
                else f"{r['pearson']*100:.2f}%"
            )
            s = (
                f"**{r['spearman']*100:.2f}%**"
                if "mft" in r["model"]
                else f"{r['spearman']*100:.2f}%"
            )
            rows2.append(
                [r["timestamp_display"], name, r["step"], p, s, r["num_samples"]]
            )

        output_lines.append(f"## Best Results - {split_title}\n")
        output_lines.append(
            format_table(
                [
                    "Timestamp",
                    "Model",
                    "Training Step",
                    "Pearson",
                    "Spearman",
                    "Samples",
                ],
                rows2,
            )
        )
        output_lines.append("\n\n")

    output_content = "\n".join(output_lines)
    print(output_content)

    with open("STS_BENCHMARK_RESULTS.md", "w", encoding="utf-8") as f:
        f.write("# STS Benchmark Results Report\n\n" + output_content)

    print("\nResults exported to STS_BENCHMARK_RESULTS.md")


if __name__ == "__main__":
    main()

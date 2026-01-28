#!/usr/bin/env python3
"""MTEB Benchmark Results Report Generator."""
import json
import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def get_model_results(base_dir="results"):
    """
    Traverse results directory to find models and their scores.
    Structure: results/<model_name>/<revision>/*.json
    """
    model_data = {}

    if not os.path.isdir(base_dir):
        print(f"Directory '{base_dir}' not found.")
        return {}

    # Iterate over models
    for model_dir_name in os.listdir(base_dir):
        model_path = os.path.join(base_dir, model_dir_name)
        if not os.path.isdir(model_path):
            continue

        clean_name = model_dir_name
        if "__" in clean_name:
            clean_name = clean_name.split("__")[-1]

        # Find revisions
        revisions = [
            d
            for d in os.listdir(model_path)
            if os.path.isdir(os.path.join(model_path, d))
        ]
        if not revisions:
            print(f"No revisions found for {model_dir_name}")
            continue

        # Pick latest
        revisions.sort(
            key=lambda x: os.path.getmtime(os.path.join(model_path, x)), reverse=True
        )
        latest_rev = revisions[0]
        rev_path = os.path.join(model_path, latest_rev)

        # Parse results
        tasks = []
        json_files = glob.glob(os.path.join(rev_path, "*.json"))

        for json_file in json_files:
            filename = os.path.basename(json_file)
            if filename == "model_meta.json":
                continue

            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                score = None

                # Try standard MTEB format
                if "scores" in data:
                    scores = data["scores"]
                    for split in [
                        "test",
                        "test_matched",
                        "test_mismatched",
                        "validation",
                        "dev",
                    ]:
                        if split in scores and scores[split]:
                            first_res = scores[split][0]
                            if "main_score" in first_res:
                                score = first_res["main_score"]
                                break

                if score is not None:
                    tasks.append(
                        {
                            "task_name": data.get(
                                "task_name", filename.replace(".json", "")
                            ),
                            "score": score * 100,  # Convert to percentage
                            "filename": filename,
                        }
                    )
            except Exception as e:
                print(f"Error reading {json_file}: {e}")

        if tasks:
            avg_score = sum(t["score"] for t in tasks) / len(tasks)
            model_data[clean_name] = {
                "revision": latest_rev,
                "full_name": model_dir_name,
                "average_score": avg_score,
                "tasks": tasks,
            }

    return model_data


def format_table(headers, rows):
    """Format data as a Markdown table."""
    if not rows:
        return "No data available."

    col_widths = []
    for i, h in enumerate(headers):
        w = len(h)
        if rows:
            max_row_w = max(len(str(r[i])) for r in rows)
            w = max(w, max_row_w)
        col_widths.append(w)

    fmt = "| " + " | ".join(f"{{:<{w}}}" for w in col_widths) + " |"
    sep = "| " + " | ".join("-" * w for w in col_widths) + " |"

    lines = [fmt.format(*headers), sep]
    for r in rows:
        lines.append(fmt.format(*[str(c) for c in r]))

    return "\n".join(lines)


def categorize_task(task_name):
    """Categorize MTEB tasks based on their names."""
    tn = task_name.lower()
    if "retrieval" in tn or "corpus" in tn or "fact" in tn:
        return "Retrieval"
    elif "clustering" in tn:
        return "Clustering"
    elif "sts" in tn:
        return "STS"
    elif "nli" in tn or "snli" in tn or "mnli" in tn:
        return "Pair Classification"
    elif "classification" in tn or "sentiment" in tn or "irony" in tn:
        return "Classification"
    elif "bitext" in tn:
        return "BitextMining"
    else:
        return "Other"


def set_academic_style():
    """Set matplotlib style for academic figures."""
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        plt.style.use("ggplot")

    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "figure.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def generate_charts(model_data):
    """Generate summary and per-task charts."""
    if not model_data:
        return

    output_files = []
    set_academic_style()

    # 1. Average Score Comparison
    # Horizontal bar chart for better readability
    fig_height = max(5, len(model_data) * 1.5)
    plt.figure(figsize=(10, fig_height))  # Fixed width, variable height

    # Sort models by average score
    sorted_models = sorted(
        model_data.items(),
        key=lambda x: x[1]["average_score"],
        reverse=False,  # Ascending for barh (top is last)
    )
    names = [m[0] for m in sorted_models]
    scores = [m[1]["average_score"] for m in sorted_models]

    # Colors: Highlight MFT vs Tabi vs Random
    colors = []
    edgecolors = []
    hatches = []

    # Define palette
    color_mft = "#4e79a7"  # Blue
    color_tabi = "#f28e2b"  # Orange
    color_other = "gray"

    for name in names:
        n_lower = name.lower()
        if "mft" in n_lower:
            colors.append(color_mft)
            edgecolors.append("black")
            hatches.append("///")
        elif "tabi" in n_lower:
            colors.append(color_tabi)
            edgecolors.append("gray")
            hatches.append("")
        else:
            colors.append(color_other)
            edgecolors.append("gray")
            hatches.append("")

    bars = plt.barh(
        names, scores, color=colors, edgecolor=edgecolors, alpha=0.9, height=0.6
    )

    # Apply hatches
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)
        if hatch:
            bar.set_linewidth(1.2)

    plt.title("Average MTEB Score Comparison", pad=15)
    plt.xlabel("Average Score (%)")

    # Valid ticks
    plt.grid(True, axis="x", linestyle="--", alpha=0.5)

    # Bold MFT labels on y-axis
    ax = plt.gca()
    for i, lbl in enumerate(ax.get_yticklabels()):
        # lbl is Text object, names[i] corresponds
        text = names[i]
        if "mft" in text.lower():
            lbl.set_fontweight("bold")
            lbl.set_color("#2c3e50")

    plt.bar_label(bars, fmt="%.2f", padding=3)
    plt.tight_layout()

    avg_chart = "mteb_average_scores.png"
    plt.savefig(avg_chart, bbox_inches="tight")
    plt.close()
    output_files.append(avg_chart)
    print(f"Generated {avg_chart}")

    return output_files


def generate_latex_table(model_data):
    """Generate detailed LaTeX table comparing all models per task."""
    output_file = "mteb_detailed_table.tex"

    if not model_data:
        return

    # Identify models
    # We want MFT first, then others sorted alphabetically
    all_models = sorted(model_data.keys())
    mft_keys = [m for m in all_models if "mft" in m.lower()]
    other_keys = [m for m in all_models if "mft" not in m.lower()]

    sorted_keys = mft_keys + other_keys

    # Shorten headers
    headers = []
    for k in sorted_keys:
        if "mft" in k.lower():
            headers.append("MFT")
        elif "tabi" in k.lower():
            headers.append("Tabi")
        elif "cosmos" in k.lower():
            headers.append("Cosmos")
        elif "mursit" in k.lower():
            headers.append("Mursit")
        else:
            headers.append(k[:6])

    # Collect all tasks
    all_tasks = set()
    for m in model_data.values():
        for t in m["tasks"]:
            all_tasks.add(t["task_name"])

    common_tasks = sorted(list(all_tasks))

    lines = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(r"\resizebox{\linewidth}{!}{")

    # Format: l + r * num_models
    col_def = "l" + "r" * len(sorted_keys)
    lines.append(f"\\begin{{tabular}}{{{col_def}}}")
    lines.append(r"\toprule")

    # Header row
    header_row = (
        r"\textbf{Task} & " + " & ".join([f"\\textbf{{{h}}}" for h in headers]) + r" \\"
    )
    lines.append(header_row)
    lines.append(r"\midrule")

    # Categories
    current_cat = ""

    # Sort by category then name
    task_objs = [{"name": t, "cat": categorize_task(t)} for t in common_tasks]
    task_objs.sort(key=lambda x: (x["cat"], x["name"]))

    for t_obj in task_objs:
        t = t_obj["name"]
        cat = t_obj["cat"]

        # Add category header if changed
        if cat != current_cat:
            lines.append(
                f"\\multicolumn{{{len(sorted_keys) + 1}}}{{c}}{{\\textit{{{cat}}}}} \\\\"
            )  # \midrule
            current_cat = cat

        # Get scores for this task across all models
        row_scores = []
        for k in sorted_keys:
            tasks = {mt["task_name"]: mt["score"] for mt in model_data[k]["tasks"]}
            row_scores.append(tasks.get(t, -1))  # -1 if missing

        max_score = max(row_scores) if row_scores else -1

        # Format row
        row_strs = []
        for s in row_scores:
            if s < 0:
                row_strs.append("-")
            else:
                s_str = f"{s:.2f}"
                if s == max_score and max_score > 0:
                    row_strs.append(f"\\textbf{{{s_str}}}")
                else:
                    row_strs.append(s_str)

        # Clean task name
        task_display = t.replace("_", r"\_")

        row_content = " & ".join(row_strs)
        lines.append(f"{task_display} & {row_content} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"}")
    lines.append(
        r"\caption{Detailed MTEB-TR performance comparison across all tasks. Best scores in bold.}"
    )
    lines.append(r"\label{tab:mteb_detailed}")
    lines.append(r"\end{table}")

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Generated {output_file}")


def main():
    print("Scanning results directory...")
    data = get_model_results()

    if not data:
        print("No model results found.")
        return

    # Prepare Markdown Report
    lines = ["# MTEB Benchmark Results Report\n"]

    # 1. Gather all tasks and all models
    all_models = sorted(data.keys())
    all_tasks = set()
    for m in data.values():
        for t in m["tasks"]:
            all_tasks.add(t["task_name"])
    all_tasks = sorted(list(all_tasks))

    # --- Table 1: All Tasks with Highlighting ---
    lines.append("# 🏆 Detailed Task Results\n")

    # Header
    table1_header = ["Task", "Category"] + all_models
    table1_rows = []

    for task in all_tasks:
        row = [task, categorize_task(task)]
        scores = []
        for model in all_models:
            # Find score for this model & task
            m_tasks = data[model]["tasks"]
            matches = [t for t in m_tasks if t["task_name"] == task]
            if matches:
                scores.append(matches[0]["score"])
            else:
                scores.append(-1.0)  # Indicator for missing

        # Determine max score (ignoring missing)
        valid_scores = [s for s in scores if s >= 0]
        max_score = max(valid_scores) if valid_scores else -1

        for s in scores:
            if s < 0:
                row.append("-")
            else:
                s_str = f"{s:.2f}%"
                if s == max_score and max_score > 0:
                    row.append(f"**{s_str}**")
                else:
                    row.append(s_str)
        table1_rows.append(row)

    lines.append(format_table(table1_header, table1_rows))
    lines.append("\n")

    # --- Table 2: Categorized Results ---
    lines.append("# 📂 Categorized Results\n")

    # Calculate average score per category per model
    categories = sorted(list(set(categorize_task(t) for t in all_tasks)))
    table2_header = ["Category"] + all_models
    table2_rows = []

    for cat in categories:
        row = [cat]
        cat_avg_scores = []

        # First pass: calculate averages
        for model in all_models:
            m_tasks = data[model]["tasks"]
            cat_scores = [
                t["score"] for t in m_tasks if categorize_task(t["task_name"]) == cat
            ]

            if cat_scores:
                avg = sum(cat_scores) / len(cat_scores)
                cat_avg_scores.append(avg)
            else:
                cat_avg_scores.append(-1.0)

        # Find max
        valid_avgs = [s for s in cat_avg_scores if s >= 0]
        max_avg = max(valid_avgs) if valid_avgs else -1

        # Second pass: format
        for avg in cat_avg_scores:
            if avg < 0:
                row.append("-")
            else:
                s_str = f"{avg:.2f}%"
                if avg == max_avg and max_avg > 0:
                    row.append(f"**{s_str}**")
                else:
                    row.append(s_str)

        table2_rows.append(row)

    lines.append(format_table(table2_header, table2_rows))
    lines.append("\n")

    # --- Table 3: Average of All (Summary) ---
    lines.append("# 📊 Overall Average Scores\n")

    sorted_models_by_avg = sorted(
        data.items(), key=lambda x: x[1]["average_score"], reverse=True
    )
    table3_header = ["Model", "Average Score", "Tasks Evaluated"]
    table3_rows = []

    best_avg = sorted_models_by_avg[0][1]["average_score"]

    for name, info in sorted_models_by_avg:
        avg_str = f"{info['average_score']:.2f}%"
        if info["average_score"] == best_avg:
            name_fmt = f"**{name}**"
            avg_str = f"**{avg_str}**"
        else:
            name_fmt = name

        table3_rows.append([name_fmt, avg_str, len(info["tasks"])])

    lines.append(format_table(table3_header, table3_rows))
    lines.append("\n")

    # Comparison Chart
    generate_charts(data)
    lines.append("![Average MTEB Scores](mteb_average_scores.png)\n")

    # Generate LaTeX Table
    generate_latex_table(data)

    # Write to file
    output_filename = "MTEB_BENCHMARK_RESULTS.md"
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Report generated: {output_filename}")


if __name__ == "__main__":
    main()

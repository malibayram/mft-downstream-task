#!/usr/bin/env python3
"""
Generate Version History Charts from Markdown Report.
Parses VERSION_BENCHMARK_RESULTS.md to extract historical performance data for random models.
"""
import re
import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


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
            "lines.linewidth": 2,
            "lines.markersize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def parse_markdown_table(file_path):
    """Parse markdown file to find model tables and extract data."""
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Define models to extract
    target_models = ["mft-random-init", "tabi-random-init"]
    data = {}

    for model in target_models:
        # Regex to find the section for the model
        # Matches "## modelname" followed by content until next "## " or EOF
        section_pattern = re.compile(
            f"## {re.escape(model)}(.*?)(?=^## |\Z)", re.MULTILINE | re.DOTALL
        )
        match = section_pattern.search(content)

        if not match:
            print(f"Warning: Section for {model} not found.")
            continue

        section_text = match.group(1)

        # Regex to parse table rows: | Date | Revision | Pearson | Spearman | ...
        # Example: | 2026-01-25 19:44 | c7dededb | 50.38% | 49.36% | ...
        row_pattern = re.compile(
            r"\|\s+(\d{4}-\d{2}-\d{2}\s\d{2}:\d{2})\s+\|\s+\w+\s+\|\s+([\d\.]+)%\s+\|\s+([\d\.]+)%\s+\|",
            re.MULTILINE,
        )

        points = []
        for row_match in row_pattern.finditer(section_text):
            date_str = row_match.group(1)
            pearson = float(row_match.group(2))
            spearman = float(row_match.group(3))

            try:
                dt = datetime.datetime.strptime(date_str, "%Y-%m-%d %H:%M")
                points.append({"date": dt, "pearson": pearson, "spearman": spearman})
            except ValueError:
                continue

        # Sort by date
        points.sort(key=lambda x: x["date"])
        data[model] = points
        print(f"Extracted {len(points)} data points for {model}")

    return data


def plot_charts(data):
    """Generate Pearson and Spearman charts."""
    set_academic_style()

    if not data:
        print("No data to plot.")
        return

    # Define colors
    model_colors = {
        "mft-random-init": "#4e79a7",  # Blue
        "tabi-random-init": "#f28e2b",  # Orange
    }
    model_markers = {"mft-random-init": "o", "tabi-random-init": "s"}

    # Helper function for plotting
    def create_plot(metric_key, title, filename):
        plt.figure(figsize=(10, 6))

        max_steps = 0

        for model in ["mft-random-init", "tabi-random-init"]:
            if model not in data:
                continue

            points = data[model]
            # Use 1-based index as step number
            steps = list(range(1, len(points) + 1))
            scores = [p[metric_key] for p in points]
            max_steps = max(max_steps, len(points))

            plt.plot(
                steps,
                scores,
                label=model,
                color=model_colors.get(model, "gray"),
                marker=model_markers.get(model, "o"),
                linestyle="-",
                linewidth=2,
            )

        plt.title(f"Version History - {title}", pad=15)
        plt.xlabel("Revision Step")
        plt.ylabel(f"{title} (%)")

        # Force integer ticks on X-axis
        plt.xlim(0.5, max_steps + 0.5)
        from matplotlib.ticker import MaxNLocator

        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

        plt.legend(frameon=True)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(filename, bbox_inches="tight")
        plt.close()
        print(f"Generated {filename}")

    create_plot("pearson", "Pearson Correlation", "version_history_pearson.png")
    create_plot("spearman", "Spearman Correlation", "version_history_spearman.png")


if __name__ == "__main__":
    file_path = "VERSION_BENCHMARK_RESULTS.md"
    data = parse_markdown_table(file_path)
    plot_charts(data)

#!/usr/bin/env python3
"""
Generate TurBLiMP Results Table in LaTeX.
Parses CSV results from turblimp_results/ and generates a comparison table.
"""
import pandas as pd
import os
import glob


def generate_latex_table():
    base_dir = "TurBLiMP/turblimp_results"

    if not os.path.exists(base_dir):
        print(f"Directory {base_dir} not found.")
        return

    # Find all sensitivity CSVs
    csv_files = glob.glob(os.path.join(base_dir, "*_turblimp_sensitivity.csv"))
    if not csv_files:
        print("No TurBLiMP result files found.")
        return

    # Map filename prefix to display name
    # e.g. cosmosGPT2-random-init -> Cosmos-Random
    #      mft-random-init -> MFT-Random
    #      tabi-random-init -> Tabi-Random
    #      newmindaiMursit-random-init -> Mursit-Random

    def get_display_name(fname):
        base = os.path.basename(fname).replace("_turblimp_sensitivity.csv", "")
        if "cosmos" in base.lower():
            return "Cosmos-Random"
        if "mursit" in base.lower():
            return "Mursit-Random"
        if "mft" in base.lower():
            return "MFT-Random"
        if "tabi" in base.lower():
            return "Tabi-Random"
        return base

    data_frames = {}
    for cf in csv_files:
        name = get_display_name(cf)
        try:
            df = pd.read_csv(cf)
            # Normalize column name if needed
            if "avg_similarity" in df.columns:
                df.rename(columns={"avg_similarity": name}, inplace=True)
                data_frames[name] = df[["category", name]]
        except Exception as e:
            print(f"Error loading {cf}: {e}")

    if not data_frames:
        return

    # Merge all dataframes on 'category'
    dfs = list(data_frames.values())
    df_merged = dfs[0]
    for df in dfs[1:]:
        df_merged = pd.merge(df_merged, df, on="category", how="outer")

    # Sort by first model's score just to have an order
    first_col = dfs[0].columns[1]  # category is 0
    df_merged.sort_values(first_col, ascending=False, inplace=True)

    # Generate LaTeX
    latex_lines = []
    latex_lines.append(r"\begin{table}[H]")
    latex_lines.append(r"\centering")
    latex_lines.append(r"\resizebox{\linewidth}{!}{")

    # Dynamic columns: Linguistic Phenomenon + Model Names
    # We don't have a single delta anymore, maybe just list them all?
    # Or keep it simple.

    model_names = [c for c in df_merged.columns if c != "category"]
    col_def = "l" + "r" * len(model_names)

    latex_lines.append(f"\\begin{{tabular}}{{{col_def}}}")
    latex_lines.append(r"\toprule")

    header = " & ".join([r"\textbf{" + m + "}" for m in model_names])
    latex_lines.append(r"\textbf{Linguistic Phenomenon} & " + header + r" \\")
    latex_lines.append(r"\midrule")

    for _, row in df_merged.iterrows():
        cat = str(row["category"]).replace("_", " ").title()

        row_str = f"{cat}"

        # Find max score for bolding
        scores = [row[m] for m in model_names if pd.notna(row[m])]
        max_score = max(scores) if scores else -1

        for m in model_names:
            val = row[m]
            if pd.isna(val):
                row_str += " & -"
            else:
                s_str = f"{val:.1%}".replace("%", r"\%")
                if val == max_score:
                    s_str = f"\\textbf{{{s_str}}}"
                row_str += f" & {s_str}"

        latex_lines.append(row_str + r" \\")

    latex_lines.append(r"\bottomrule")
    latex_lines.append(r"\end{tabular}")
    latex_lines.append(r"}")
    latex_lines.append(
        r"\caption{Detailed TurBLiMP sensitivity scores comparison across all models.}"
    )
    latex_lines.append(r"\label{tab:turblimp_detailed}")
    latex_lines.append(r"\end{table}")

    output_filename = "turblimp_table.tex"
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write("\n".join(latex_lines))

    print(f"Generated {output_filename}")


if __name__ == "__main__":
    generate_latex_table()

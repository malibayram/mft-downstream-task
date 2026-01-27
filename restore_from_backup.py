import re
import os


def extract_section(content, section_name, next_section_name):
    pattern = re.compile(
        r"\\section\{"
        + re.escape(section_name)
        + r"\}(.*?)\\section\{"
        + re.escape(next_section_name)
        + r"\}",
        re.DOTALL,
    )
    match = pattern.search(content)
    if match:
        return f"\\section{{{section_name}}}\n{match.group(1).strip()}\n"
    return None


def extract_last_section(content, section_name, end_marker):
    pattern = re.compile(
        r"\\section\{" + re.escape(section_name) + r"\}(.*?)" + re.escape(end_marker),
        re.DOTALL,
    )
    match = pattern.search(content)
    if match:
        return f"\\section{{{section_name}}}\n{match.group(1).strip()}\n"
    return None


backup_path = (
    "Tokens_with_Meaning__A_Hybrid_Tokenization_Approach_for_NLP/main_backup.tex"
)
base_dir = "Tokens_with_Meaning__A_Hybrid_Tokenization_Approach_for_NLP"

with open(backup_path, "r") as f:
    content = f.read()

# Introduction
intro = extract_section(content, "Introduction", "Related Work")
if intro:
    with open(os.path.join(base_dir, "introduction.tex"), "w") as f:
        f.write(intro)
    print("Restored introduction.tex")
else:
    print("Could not find Introduction")

# Related Work
related = extract_section(content, "Related Work", "Methodology")
if related:
    with open(os.path.join(base_dir, "related_work.tex"), "w") as f:
        f.write(related)
    print("Restored related_work.tex")
else:
    print("Could not find Related Work")

# Methodology
method = extract_section(content, "Methodology", "Results and Analysis")
if method:
    with open(os.path.join(base_dir, "methodology.tex"), "w") as f:
        f.write(method)
    print("Restored methodology.tex")
else:
    print("Could not find Methodology")

# Results and Analysis
# Ends at \\section{Conclusion}
results = extract_last_section(content, "Results and Analysis", "\\section{Conclusion}")
if results:
    with open(os.path.join(base_dir, "results_and_analysis.tex"), "w") as f:
        f.write(results)
    print("Restored results_and_analysis.tex")
else:
    print("Could not find Results and Analysis")

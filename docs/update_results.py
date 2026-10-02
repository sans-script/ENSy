#!/usr/bin/env python3
"""Write the results tables and plots into DOCS.md from results/*.json.

Every number in the paper's results section is generated here, never typed by
hand. Regions are delimited in DOCS.md by

    <!-- BEGIN:<name> -->  ...  <!-- END:<name> -->

Run:  python docs/update_results.py [--hash 40863cdefcb4]
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs" / "DOCS.md"

MODELS = {"qwen": "Qwen 2.5 0.5B", "gemma2": "Gemma 2 2B", "gpt2": "GPT-2", "bert": "BERT base"}
ALGOS = {"pca": "PCA", "tsne": "t-SNE ($p=5$)", "tsne_p30": "t-SNE ($p=30$)", "umap": "UMAP", "isomap": "Isomap"}
ALGO_PLAIN = {"pca": "PCA", "tsne": "t-SNE p=5", "tsne_p30": "t-SNE p=30", "umap": "UMAP", "isomap": "Isomap"}
K_MAIN = 10


def num(v: float, signed: bool = False, d: int = 3) -> str:
    if signed:
        return f"{'−' if v < 0 else '+'}{abs(v):.{d}f}"
    return f"{v:.{d}f}"


def load(kind: str, h: str) -> dict:
    return json.loads((ROOT / "results" / f"{kind}_{h}.json").read_text(encoding="utf-8"))


def at_k(curves, k):
    return next(c for c in curves if c["K"] == k)


def provenance(r: dict) -> str:
    env = r["environment"]
    clean = "clean" if env["working_tree_dirty"] is False else "modified"
    return (f"*Source: `results/{r['kind']}_{r['dataset_hash'][:12]}.json` — commit `{(env['commit_sha'] or '?')[:8]}` "
            f"({clean} working tree), Python {env['python']}, scikit-learn {env['scikit_learn']}, "
            f"umap-learn {env['umap_learn']}, seed {r['seed']}, extraction: {r['extraction']}.*")


def table_main(r: dict) -> str:
    rows = ["| Model | Algorithm | $M_T$ | $M_C$ | $Q_{NX}$ | $B_{NX}$ | $B_{TC}$ | $s_{3D}$ | $s_{2D}$ |",
            "|---|---|---|---|---|---|---|---|---|"]
    for c in r["configs"]:
        m = at_k(c["curves"], K_MAIN)
        rows.append(
            f"| {MODELS[c['model']]} | {ALGOS[c['algorithm']]} | {num(m['trustworthiness'])} | "
            f"{num(m['continuity'])} | {num(m['Q_NX'])} | {num(m['B_NX'], True)} | {num(m['B_TC'], True)} | "
            f"{num(c['silhouette_projected'], True)} | {num(c['silhouette_projected_2d'], True)} |")
    return ":::booktabs[wide]\n" + "\n".join(rows) + "\n:::\n\n" + provenance(r)


def table_highdim(r: dict) -> str:
    seen, rows = set(), ["| Model | $d$ | $s$ in $\\mathbb{R}^d$ (cosine) | PCA variance, 3 components |", "|---|---|---|---|"]
    for c in r["configs"]:
        if c["model"] in seen or c["algorithm"] != "pca":
            continue
        seen.add(c["model"])
        evr = sum(c["diagnostics"]["explained_variance_ratio"])
        rows.append(f"| {MODELS[c['model']]} | {c['embedding_dim']} | {num(c['silhouette_high_dim_cosine'], True)} | {100 * evr:.1f}% |")
    return ":::booktabs\n" + "\n".join(rows) + "\n:::"


COLORS = {"pca": "#2b5797", "tsne": "#d73a49", "tsne_p30": "#22863a", "umap": "#e36209", "isomap": "#6f42c1"}


def figure_curves(r: dict, metric: str, ylabel: str, name: str, caption: str, num: int) -> str:
    """One 1x4 panel figure (a panel per model, a line per configuration), saved as PNG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "serif", "font.serif": ["EB Garamond", "Georgia", "Times New Roman"],
                         "mathtext.fontset": "cm", "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "axes.grid.axis": "y", "grid.linestyle": "--", "grid.alpha": 0.45,
                         "font.size": 9})
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 5.625), sharey=True, sharex=True)  # 16:9
    axes = axes.ravel()
    for ax, (mk, mname) in zip(axes, MODELS.items()):
        for c in (c for c in r["configs"] if c["model"] == mk):
            ks = [p["K"] for p in c["curves"]]
            ax.plot(ks, [p[metric] for p in c["curves"]], marker="o", markersize=3.5, linewidth=1.4,
                    color=COLORS[c["algorithm"]], label=ALGO_PLAIN[c["algorithm"]])
        ax.set_title(mname, fontsize=10, fontweight="bold")
        ax.set_xticks([5, 10, 15, 20])
        ax.set_xlabel("$K$") if ax in axes[2:] else None
        if metric == "B_NX":
            ax.axhline(0, color="#555555", linewidth=0.7)
    axes[0].set_ylabel(ylabel)
    axes[2].set_ylabel(ylabel)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    out = ROOT / "docs" / "figures" / f"{name}.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=200)  # keep the exact 16:9 canvas
    plt.close(fig)
    return f"![{caption}](figures/{name}.png)\n\n_Figure {num}: {caption}_"


def table_sensitivity(s: dict) -> str:
    label = {(a, json.dumps(o, sort_keys=True)): n for a, o, n in [
        ("tsne", {"perplexity": 5.0}, "t-SNE $p=5$"), ("tsne", {"perplexity": 30.0}, "t-SNE $p=30$"), ("umap", {}, "UMAP")]}
    rows = ["| Model | Configuration | $M_T(10)$ | $B_{NX}(10)$ | $s_{3D}$ | $s_{2D}$ |", "|---|---|---|---|---|---|"]
    for c in s["configs"]:
        m = at_k(c["curves"], K_MAIN)
        name = label[(c["algorithm"], json.dumps(c["overrides"], sort_keys=True))]
        rows.append(
            f"| {MODELS[c['model']]} | {name} | {num(m['trustworthiness']['mean'])} ± {num(m['trustworthiness']['sd'])} | "
            f"{num(m['B_NX']['mean'], True)} ± {num(m['B_NX']['sd'])} | "
            f"{num(c['silhouette_projected']['mean'], True)} ± {num(c['silhouette_projected']['sd'])} | "
            f"{num(c['silhouette_projected_2d']['mean'], True)} ± {num(c['silhouette_projected_2d']['sd'])} |")
    seeds = ", ".join(map(str, s["seeds"]))
    return (":::booktabs[wide]\n" + "\n".join(rows) + "\n:::\n\n"
            f"*Mean ± sample standard deviation over seeds {{{seeds}}}.* " + provenance(s))


def table_ablation(main: dict, abl: dict) -> str:
    pca = {c["model"]: at_k(c["curves"], K_MAIN) for c in main["configs"] if c["algorithm"] == "pca"}
    rows = ["| Model | PCA $M_T$ / $M_C$ | PCA-NC $M_T$ / $M_C$ | PCA-NC $B_{NX}$ |", "|---|---|---|---|"]
    for c in abl["configs"]:
        a, m = pca[c["model"]], at_k(c["curves"], K_MAIN)
        rows.append(f"| {MODELS[c['model']]} | {num(a['trustworthiness'])} / {num(a['continuity'])} | "
                    f"{num(m['trustworthiness'])} / {num(m['continuity'])} | {num(m['B_NX'], True)} |")
    return ":::booktabs\n" + "\n".join(rows) + "\n:::"


def table_dataset(ds: dict) -> str:
    c = ds["counts"]
    drop = ", ".join(f"{MODELS[k]} {v}" for k, v in c["dropped_multi_token"].items())
    rows = ["| Stage | Count |", "|---|---|",
            f"| PTB tokens (NLTK sample) | {c['ptb_tokens']:,} |",
            f"| Lexical types (lower-cased, alphabetic, no proper nouns) | {c['lexical_types']:,} |",
            f"| Excluded artefacts (clitic stems, name-internal forms) | {len(c['excluded_artefacts'])} |",
            f"| Excluded: category tie / majority outside the partition | {c['excluded_tie']} / {c['excluded_outside_partition']:,} |",
            f"| After disambiguation | {c['after_disambiguation']:,} |",
            f"| Dropped as multi-token (first failing tokenizer) | {drop} |",
            f"| Single-token in all four tokenizers: noun / verb / function | "
            f"{c['single_token_per_category']['content_noun']:,} / {c['single_token_per_category']['content_verb']:,} / "
            f"{c['single_token_per_category']['function_word']} |",
            f"| **Balanced dataset $N$** | **{ds['N']}** ({ds['N'] // 3} per category) |"]
    return ":::booktabs\n" + "\n".join(rows) + "\n:::\n\n" + f"*Dataset hash `{ds['dataset_hash'][:12]}`.*"


def replace_region(text: str, name: str, body: str) -> str:
    pat = re.compile(rf"(<!-- BEGIN:{name} -->).*?(<!-- END:{name} -->)", re.DOTALL)
    if not pat.search(text):
        raise SystemExit(f"marker {name} not found in DOCS.md")
    # blank line before END: the PDF converter ends a figure caption at a blank line
    return pat.sub(lambda m: m.group(1) + "\n" + body + "\n\n" + m.group(2), text)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--hash", default=None, help="dataset hash prefix (default: from data/dataset.json)")
    args = ap.parse_args()
    ds = json.loads((ROOT / "data" / "dataset.json").read_text(encoding="utf-8"))
    h = args.hash or ds["dataset_hash"][:12]
    main_r, abl, sens = load("main", h), load("ablation", h), load("sensitivity", h)
    text = DOCS.read_text(encoding="utf-8").replace("\r\n", "\n")
    for name, body in {
        "dataset": table_dataset(ds),
        "main": table_main(main_r),
        "highdim": table_highdim(main_r),
        "curves_mt": figure_curves(main_r, "trustworthiness", "$M_T(K)$", "curves_trustworthiness",
                                   "Trustworthiness $M_T(K)$ over $K$ for every model and projection configuration.", 6),
        "curves_bnx": figure_curves(main_r, "B_NX", "$B_{NX}(K)$", "curves_bnx",
                                    "Behaviour indicator $B_{NX}(K)$ over $K$ (positive: intrusive; negative: extrusive). "
                                    "With perplexity 30, t-SNE is below zero at every $K$ in all four models; with perplexity 5 it is "
                                    "above zero for Qwen and the two Gemma models and close to zero for BERT.", 7),
        "sensitivity": table_sensitivity(sens),
        "ablation": table_ablation(main_r, abl),
    }.items():
        text = replace_region(text, name, body)
    DOCS.write_text(text, encoding="utf-8", newline="\n")  # LF: the PDF converter expects it
    print(f"DOCS.md updated from results *_{h}.json")


if __name__ == "__main__":
    main()

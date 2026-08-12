"""Generate figures and RESULTS.md from whatever experiments have actually run.

Every table states its exact n. Missing experiments are reported as missing
rather than omitted, so the report can never imply coverage that does not exist.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

RESULTS = Path("results")
FIGURES = RESULTS / "figures"

PALETTE = {
    "max-eig": "#2E6F9E",
    "tool-eig-per-cost": "#2E6F9E",
    "greedy-frequency": "#D97A34",
    "patient-only": "#D97A34",
    "random": "#9AA0A6",
    "random-tool": "#9AA0A6",
    "cheapest-first": "#6C8E68",
    "prior-only": "#B04A4A",
}


def _style():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.5,
        "legend.frameon": False,
    })
    return plt


def load(name: str):
    path = RESULTS / name
    return json.loads(path.read_text()) if path.exists() else None


def fig_budget_sweep(plt) -> str | None:
    data = load("budget_sweep.json")
    if not data:
        return None
    settings = data["settings"]
    fig, axes = plt.subplots(1, len(settings), figsize=(4.2 * len(settings), 3.4),
                             sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (name, block) in zip(axes, settings.items()):
        for pname, d in block["policies"].items():
            acc = np.array(d["accuracy"]) * 100
            ax.plot(np.arange(len(acc)), acc, label=pname,
                    color=PALETTE.get(pname, "#444"), linewidth=1.8)
        ax.set_title(f"{name}\n(noise={block['config']['env_noise']}, "
                     f"omission={block['config']['omission']})", fontsize=9)
        ax.set_xlabel("questions asked")
        # Question counts are integers; matplotlib's default 2.5 ticks are wrong here.
        n_q = len(next(iter(block["policies"].values()))["accuracy"]) - 1
        ax.set_xticks(np.arange(0, n_q + 1, max(1, n_q // 5)))
    axes[0].set_ylabel("top-1 accuracy (%)")
    axes[-1].legend(loc="upper left", fontsize=8)
    fig.suptitle(f"Accuracy vs question budget  (n={data['n_cases']} cases, "
                 f"{data['n_conditions']} conditions)", fontsize=10)
    fig.tight_layout()
    out = FIGURES / "budget_sweep.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return str(out)


def fig_calibration(plt) -> str | None:
    data = load("calibration.json")
    if not data:
        return None
    unc, cal = data["test_uncalibrated"], data["test_calibrated"]
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.3))

    ax = axes[0]
    keys = ["ece", "brier"]
    x = np.arange(len(keys))
    ax.bar(x - 0.2, [unc[k] for k in keys], 0.4, label="uncalibrated", color="#B04A4A")
    ax.bar(x + 0.2, [cal[k] for k in keys], 0.4, label="calibrated", color="#2E6F9E")
    ax.set_xticks(x)
    ax.set_xticklabels(["ECE", "Brier"])
    ax.set_title("Calibration error (lower is better)", fontsize=9)
    ax.legend(fontsize=8)

    ax = axes[1]
    keys = ["accuracy_selective", "selective_risk", "coverage"]
    x = np.arange(len(keys))
    ax.bar(x - 0.2, [unc[k] for k in keys], 0.4, label="uncalibrated", color="#B04A4A")
    ax.bar(x + 0.2, [cal[k] for k in keys], 0.4, label="calibrated", color="#2E6F9E")
    ax.set_xticks(x)
    ax.set_xticklabels(["sel. accuracy", "sel. risk", "coverage"], fontsize=8)
    ax.set_title("Selective prediction", fontsize=9)

    fitted = data["fitted"]
    fig.suptitle(
        f"Calibration under misspecification  (T={fitted['temperature']}, "
        f"tau={fitted['tau']}, effective noise MLE={fitted['effective_noise_mle']} "
        f"vs true {fitted['true_env_noise']})", fontsize=9)
    fig.tight_layout()
    out = FIGURES / "calibration.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return str(out)


def fig_tool_mix(plt) -> str | None:
    data = load("tools_wellspec.json") or load("tools_pilot.json")
    if not data:
        return None
    rows = data["results"]
    kinds = sorted({k for r in rows for k in r["tool_mix"]})
    colors = {"ask_patient": "#D97A34", "query_kg": "#2E6F9E",
              "retrieve_text": "#6C8E68", "verify_herb": "#8E6C9E"}

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4))
    ax = axes[0]
    names = [r["agent"] for r in rows]
    bottom = np.zeros(len(rows))
    for k in kinds:
        vals = np.array([r["tool_mix"].get(k, 0.0) for r in rows])
        ax.barh(names, vals, left=bottom, label=k, color=colors.get(k, "#888"))
        bottom += vals
    ax.set_xlabel("fraction of actions")
    ax.set_title("Which source the agent consults", fontsize=9)
    ax.legend(fontsize=7, loc="lower right")
    ax.invert_yaxis()

    # Plot against *patient questions*, not total cost. Every agent spends its
    # whole budget, so total cost is ~constant by construction and a
    # "cost-accuracy frontier" would be a vertical line. Patient burden is the
    # axis that actually varies, and it is the one a clinician cares about.
    ax = axes[1]
    ax.scatter([r["mean_patient_questions"] for r in rows],
               [100 * r["accuracy"] for r in rows],
               s=52, color="#2E6F9E", zorder=3)
    for r in rows:
        ax.annotate(r["agent"].split("-", 1)[0],
                    (r["mean_patient_questions"], 100 * r["accuracy"]),
                    textcoords="offset points", xytext=(6, 3), fontsize=7)
    ax.set_xlabel("patient questions asked (burden)")
    ax.set_ylabel("accuracy (%)")
    ax.set_title("Accuracy vs patient burden\n(equal total cost budget)", fontsize=9)
    ax.set_xlim(-1.5, 17)
    ax.set_ylim(-5, 105)

    fig.suptitle(f"Heterogeneous action space  (n={rows[0]['n']} cases, "
                 f"{data['action_space']['n_actions']} actions)", fontsize=10)
    fig.tight_layout()
    out = FIGURES / "tool_selection.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return str(out)


def fig_agent_comparison(plt) -> str | None:
    made = []
    for stage in ("structured", "structured-misspecified", "pilot"):
        data = load(f"evaluation_{stage}.json")
        if not data:
            continue
        rows = [r for r in data["results"] if r["agent"] != "B1-full-information"]
        rows.sort(key=lambda r: r["accuracy_overall"])
        fig, ax = plt.subplots(figsize=(6.2, 3.2))
        names = [r["agent"] for r in rows]
        vals = [100 * r["accuracy_overall"] for r in rows]
        colors = ["#2E6F9E" if "eig" in n else "#9AA0A6" for n in names]
        ax.barh(names, vals, color=colors)
        for i, v in enumerate(vals):
            ax.text(v + 0.6, i, f"{v:.1f}%", va="center", fontsize=8)
        ax.set_xlabel("top-1 accuracy (%)")
        n = list(data["cases_per_agent"].values())[0]
        ax.set_title(f"{stage}  (n={n} cases, {data['n_conditions']} conditions)",
                     fontsize=9)
        ax.set_xlim(0, max(vals) * 1.18)
        fig.tight_layout()
        out = FIGURES / f"agents_{stage}.png"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        made.append(str(out))
    return made[0] if made else None


def format_cell(key: str, v) -> str:
    """Format by column meaning: percentages, p-values and counts each differ."""
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if not isinstance(v, float):
        return str(v)
    if key == "accuracy_per_cost":
        return f"{v:.4f}"
    if (key.startswith(("acc", "cov", "hit"))
            or key.endswith(("_rate", "_accuracy"))):
        return f"{100 * v:.1f}%"
    if key.startswith("p_") or key.startswith("p ("):
        # Never print a p-value as 0.000 - report the bound instead.
        return "<1e-16" if v < 1e-16 else f"{v:.2e}"
    if key.startswith("delta"):
        return f"{v:+.3f}"
    if key.endswith(("questions", "_n", "seconds")) or key in ("n", "mean_questions"):
        return f"{v:.1f}"
    return f"{v:.3f}"


def markdown_table(rows: list[dict], columns: list[tuple[str, str]]) -> str:
    head = "| " + " | ".join(c[1] for c in columns) + " |"
    sep = "|" + "|".join("---" for _ in columns) + "|"
    lines = [head, sep]
    for r in rows:
        lines.append("| " + " | ".join(
            format_cell(key, r.get(key)) for key, _ in columns) + " |")
    return "\n".join(lines)


def build_report() -> str:
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt = _style()

    figures = {
        "budget sweep": fig_budget_sweep(plt),
        "calibration": fig_calibration(plt),
        "tool selection": fig_tool_mix(plt),
        "agent comparison": fig_agent_comparison(plt),
    }

    env = load("environment.json") or {}
    parts: list[str] = []
    parts.append("# Results\n")
    parts.append(
        f"Generated from real runs on {env.get('chip', 'unknown')} / "
        f"{env.get('memory_gb', '?')} GB unified memory, macOS "
        f"{env.get('os_version', '?')}, Python {env.get('python_version', '?')}.\n"
    )
    parts.append(
        "Every table states its exact sample size. Experiments that have not been "
        "run are listed as missing rather than omitted.\n"
    )

    # --- main agent tables ---
    for stage in ("structured", "structured-misspecified", "pilot", "smoke"):
        data = load(f"evaluation_{stage}.json")
        if not data:
            parts.append(f"\n## {stage}\n\n_Not run._\n")
            continue
        n = list(data["cases_per_agent"].values())[0]
        parts.append(f"\n## Single-source planner - {stage}\n")
        if data.get("incomplete"):
            parts.append(f"\n> **Incomplete run** ({data.get('reason')}). "
                         "Figures below cover only the cases that finished.\n")
        parts.append(f"\nn = {n} cases, {data['n_conditions']} conditions, "
                     f"{data['n_features']} features.\n\n")
        parts.append(markdown_table(
            sorted(data["results"], key=lambda r: -r["accuracy_overall"]),
            [("agent", "agent"), ("n", "n"), ("coverage", "coverage"),
             ("accuracy_overall", "accuracy"), ("accuracy_selective", "sel. acc"),
             ("macro_f1", "macro-F1"), ("mean_questions", "questions"),
             ("ece", "ECE"), ("aurc", "AURC")]))
        parts.append("\n")
        if data.get("comparisons"):
            parts.append("\nPaired comparisons vs `B7-max-eig` "
                         "(McNemar, Holm-corrected):\n\n")
            parts.append(markdown_table(
                sorted(data["comparisons"], key=lambda c: -c["delta_accuracy"]),
                [("agent", "agent"), ("n_paired", "n"), ("delta_accuracy", "delta"),
                 ("p_mcnemar_holm", "p (Holm)"), ("significant_005", "sig.")]))
            parts.append("\n")

    # --- tools ---
    tools = load("tools_wellspec.json") or load("tools_pilot.json")
    parts.append("\n## Heterogeneous action space\n")
    if not tools:
        parts.append("\n_Not run._\n")
    else:
        a = tools["action_space"]
        parts.append(f"\nn = {tools['results'][0]['n']} cases. "
                     f"{a['n_actions']} actions: {a['by_kind']}. "
                     f"Costs: {a['costs']}.\n\n")
        parts.append(markdown_table(
            sorted(tools["results"], key=lambda r: -r["accuracy"]),
            [("agent", "agent"), ("n", "n"), ("accuracy", "accuracy"),
             ("mean_cost", "cost"), ("mean_patient_questions", "patient Qs"),
             ("accuracy_per_cost", "acc/cost"), ("ece", "ECE")]))
        parts.append("\n")
        if tools.get("comparisons"):
            parts.append("\nPaired vs `T1-tool-eig-per-cost` "
                         "(McNemar, Holm-corrected):\n\n")
            parts.append(markdown_table(
                sorted(tools["comparisons"], key=lambda c: -c["delta_accuracy"]),
                [("agent", "agent"), ("delta_accuracy", "delta"),
                 ("p_holm", "p (Holm)"), ("significant", "sig.")]))
            parts.append("\n")
        if tools.get("tool_selection_agreement_vs_oracle"):
            parts.append(f"\nTool-kind agreement with an EIG/cost oracle: "
                         f"`{tools['tool_selection_agreement_vs_oracle']}`\n")

    mis = load("tools_misspec.json")
    if mis:
        parts.append("\n### Heterogeneous action space - misspecified\n")
        c = mis["config"]
        parts.append(f"\nn = {mis['results'][0]['n']} cases, env noise {c['env_noise']}, "
                     f"omission {c['omission']}, agent assumes {c['assumed_noise']}.\n\n")
        parts.append(markdown_table(
            sorted(mis["results"], key=lambda r: -r["accuracy"]),
            [("agent", "agent"), ("n", "n"), ("accuracy", "accuracy"),
             ("mean_cost", "cost"), ("mean_patient_questions", "patient Qs"),
             ("accuracy_per_cost", "acc/cost"), ("ece", "ECE")]))
        parts.append("\n")

    # --- cost sensitivity ---
    cs = load("cost_sweep.json")
    parts.append("\n### Cost-table sensitivity\n")
    if not cs:
        parts.append("\n_Not run._\n")
    else:
        s = cs["summary"]
        parts.append(
            f"\nThe cost table is a design choice, so the advantage is re-measured across "
            f"cost ratios. `ratio` prices every non-patient channel relative to a patient "
            f"question fixed at 1.0. n = {cs['rows'][0]['n']} cases per point.\n\n")
        parts.append(markdown_table(cs["rows"], [
            ("cost_ratio", "ratio"), ("tool_eig_accuracy", "tool-EIG"),
            ("patient_only_accuracy", "patient-only"), ("delta", "delta"),
            ("p_mcnemar", "p"), ("non_patient_action_share", "tool share")]))
        parts.append(
            f"\n\nAdvantage significant and positive at "
            f"**{s['n_points_significant_positive']}/{s['n_points']}** cost ratios; "
            f"delta range {s['min_delta']:+.3f} to {s['max_delta']:+.3f}. "
            f"Positive at every ratio below 1.0: **{s['all_cheap_ratios_positive']}**.\n")

    # --- matched-observation control ---
    mb = load("matched_budget.json")
    parts.append("\n### Matched-observation control — is the advantage selective or economic?\n")
    if not mb:
        parts.append("\n_Not run._\n")
    else:
        parts.append(
            f"\nEvery agent takes exactly N observations chosen by pure EIG with **cost "
            f"ignored**; the only difference is whether non-patient channels are available. "
            f"n = {mb['rows'][0]['n']} cases per budget.\n\n")
        parts.append(markdown_table(mb["rows"], [
            ("budget_actions", "observations"),
            ("all_sources_accuracy", "all sources"),
            ("patient_only_accuracy", "patient-only"),
            ("delta", "delta"), ("p_mcnemar", "p"),
            ("non_patient_share", "tool share")]))
        s = mb["summary"]
        parts.append(
            f"\n\n**Significant and positive at {s['n_significant_positive']} of "
            f"{s['n_budgets']} budgets; mean delta {s['mean_delta']:+.3f}.**\n"
            f"\n> **The multi-source advantage is economic, not selective.** With cost "
            f"removed there is no reason to consult a lower-fidelity channel (patient "
            f"noise 0.05 vs retrieval 0.25), so spending every observation on the patient "
            f"is optimal — and the multi-source agent, still allocating 25–77% of actions "
            f"to tools, is neutral-to-worse. A heterogeneous action space is only rational "
            f"under a cost constraint. Report the equal-cost result as *cheaper "
            f"consultations at equal accuracy*, never as *better source selection*.\n")

    # --- nosology mapping ---
    nos = load("nosology_mapping.json")
    parts.append("\n## Sanskrit→English nosology mapping\n")
    if not nos:
        parts.append("\n_Not built._\n")
    else:
        parts.append(
            f"\n{nos['n_mapped_terms']} curated correspondences "
            f"({nos['by_confidence']}), unlocking **{nos['edges_unlocked']} of "
            f"{nos['total_herb_indication_edges']}** herb→indication edges "
            f"({100*nos['edge_coverage']:.1f}%).\n")
        parts.append(
            f"\n> `expert_reviewed = {nos['expert_reviewed']}`. Compiled from standard "
            f"textbook correspondences **without** a domain expert; the "
            f"`approximate` tier especially needs practitioner review before "
            f"publication.\n")

    # --- retrieval ---
    ret = load("retrieval.json")
    parts.append("\n## Text retrieval (BM25)\n")
    if not ret:
        parts.append("\n_Not built._\n")
    else:
        parts.append(f"\n{ret['n_passages']} local passages ({ret['by_source']}), "
                     f"{ret['passages_with_condition_mention']} naming a condition. "
                     f"Backend `{ret['backend']}`.\n\n")
        parts.append(f"- precision **{ret['precision']}** vs curated base rate "
                     f"{ret['curated_base_rate']} -> **{ret['lift_over_base_rate']}x lift**\n")
        parts.append(f"- recall {ret['recall']}, F1 {ret['f1']}, "
                     f"feature coverage {100*ret['mean_feature_coverage']:.1f}%\n")
        parts.append(
            f"\n> **Do not quote raw agreement.** It reads "
            f"{ret['raw_agreement_UNINFORMATIVE']}, but both matrices are ~97.5% zeros "
            f"and an all-zeros matrix scores {ret['all_zeros_would_score']} - i.e. the "
            f"text-derived matrix is *worse than predicting nothing* on that metric. "
            f"The corpus carries real but thin signal: ~3x chance precision at 6% recall.\n")

    # --- calibration ---
    cal = load("calibration.json")
    parts.append("\n## Calibration under misspecification\n")
    if not cal:
        parts.append("\n_Not run._\n")
    else:
        f = cal["fitted"]
        parts.append(
            f"\nFitted on {cal['config']['n_calib']} calibration cases, evaluated on "
            f"{cal['config']['n_test']} disjoint test cases. "
            f"Temperature {f['temperature']}, tau {f['tau']}. "
            f"Effective-noise MLE **{f['effective_noise_mle']}** against a true "
            f"environment noise of **{f['true_env_noise']}**, with the agent told "
            f"{f['assumed_noise']}.\n\n")
        rows = [{"metric": k, "uncalibrated": cal["test_uncalibrated"][k],
                 "calibrated": cal["test_calibrated"][k]}
                for k in ("coverage", "accuracy_selective", "selective_risk",
                          "ece", "brier", "aurc")]
        parts.append(markdown_table(
            rows, [("metric", "metric"), ("uncalibrated", "uncalibrated"),
                   ("calibrated", "calibrated")]))
        parts.append(
            "\n\n> AURC is unchanged by design: temperature scaling is monotonic, so "
            "it repairs calibration but cannot improve discrimination.\n")

    # --- LLM planner baseline ---
    llm = load("llm_planner.json")
    parts.append("\n## B6 - LLM as planner\n")
    if not llm:
        parts.append("\n_Not run._\n")
    else:
        parts.append(
            f"\nn = {llm['n']} cases, budget {llm['budget']}, shortlist "
            f"{llm['shortlist_size']}. **Identical posterior and identical patient "
            f"answers** - only action selection differs, so the gap is attributable "
            f"to planning alone.\n\n")
        rows = [
            {"planner": "B6 LLM (Qwen3-4B-4bit)", "accuracy": llm["llm_accuracy"],
             "seconds_per_case": llm["llm_seconds_per_case"]},
            {"planner": "B7 closed-form EIG", "accuracy": llm["eig_accuracy"],
             "seconds_per_case": llm["eig_seconds_per_case"]},
        ]
        parts.append(markdown_table(rows, [("planner", "planner"),
                                           ("accuracy", "accuracy"),
                                           ("seconds_per_case", "s/case")]))
        parts.append(
            f"\n\n- delta (EIG − LLM) **{llm['delta_eig_minus_llm']:+.3f}**, "
            f"95% CI {llm['ci95']}, McNemar p = {llm['p_mcnemar']:.2e}\n"
            f"- EIG is **{llm['speedup']}x faster** per case\n"
            f"- {100*llm['unparsed_rate']:.1f}% of {llm['llm_calls']} LLM replies could "
            f"not be parsed to a choice (fell back to the first option)\n")

    # --- BhashaBench ---
    bb = load("bhashabench.json")
    parts.append("\n## BhashaBench-Ayur\n")
    if not bb:
        parts.append("\n_Not run._\n")
    else:
        rows = []
        for lang in ("en", "hi"):
            if lang in bb:
                s = bb[lang]
                rows.append({"language": lang, "n": s["n"],
                             "accuracy": s["accuracy"],
                             "accuracy_parsed_only": s.get("accuracy_parsed_only"),
                             "chance_corrected_accuracy":
                                 s.get("chance_corrected_accuracy"),
                             "unparsed_rate": s["unparsed_rate"],
                             "seconds": s["mean_seconds"]})
        parts.append("\nFrozen Qwen3-4B-4bit, zero-shot, domain-stratified subset. "
                     "Chance level is 25% (four options).\n\n")
        parts.append(markdown_table(rows, [
            ("language", "language"), ("n", "n"), ("accuracy", "raw acc"),
            ("accuracy_parsed_only", "parsed-only"),
            ("chance_corrected_accuracy", "chance-corrected"),
            ("unparsed_rate", "unparsed"), ("seconds", "s/q")]))
        gap = bb.get("language_gap")
        if gap:
            parts.append(
                f"\n\n| gap measure | value |\n|---|---|\n"
                f"| domain-stratified ({gap['n_shared_domains']} shared domains) | "
                f"{100*gap['weighted_delta_en_minus_hi']:+.1f} pts |\n"
                f"| parsed-only | {100*gap['delta_parsed_only']:+.1f} pts |\n"
                f"| attributable to instruction-following | "
                f"{100*gap['delta_attributable_to_format']:+.1f} pts |\n"
                f"| **chance-corrected** | "
                f"**{100*gap['delta_chance_corrected']:+.1f} pts** |\n")
            parts.append(
                f"\n> **Report the chance-corrected gap.** The 25% MCQ floor compresses "
                f"both raw scores toward each other; corrected for it the gap widens to "
                f"{100*gap['delta_chance_corrected']:+.1f} points and Hindi sits at "
                f"{100*bb['hi']['chance_corrected_accuracy']:.1f}% — barely above guessing. "
                f"Only {100*gap['delta_attributable_to_format']:.1f} points come from the "
                f"model failing to emit a letter in Hindi, so the deficit is knowledge, "
                f"not formatting.\n"
                f">\n"
                f"> The naive difference of means "
                f"({100*gap['naive_delta_do_not_report']:+.1f} pts) **must not be "
                f"reported**: the two splits share zero item ids and have per-domain size "
                f"ratios from 0.36 to 1.04, so it confounds language with topic mix "
                f"(`DATA_AUDIT.md` §1).\n")

    # --- kg ---
    kg = load("kg_stats.json")
    parts.append("\n## Knowledge graph\n")
    if not kg:
        parts.append("\n_Not built._\n")
    else:
        t = kg["two_hop_evaluation"]
        parts.append(f"\n{kg['n_triples']} triples from {list(kg['sources'])}. "
                     f"AyurKOSH available: {kg['ayurkosh_available']}.\n\n")
        parts.append(f"Two-hop `condition -> dosha -> herb`, evaluated against held-out "
                     f"direct `condition -> herb` edges over "
                     f"{t['n_conditions_evaluated']} conditions and "
                     f"{t['n_candidate_herbs']} candidate herbs:\n\n")
        parts.append(f"- Hit@1 {100*t['hit@1']:.1f}%, Hit@5 {100*t['hit@5']:.1f}%, "
                     f"Hit@10 **{100*t['hit@10']:.1f}%**, Hit@20 {100*t['hit@20']:.1f}%\n")
        parts.append(f"- random Hit@10 baseline {100*t['random_hit@10_baseline']:.1f}%, "
                     f"MRR {t['mean_reciprocal_rank']}, median rank {t['median_rank']}\n")

    # --- figures ---
    parts.append("\n## Figures\n\n")
    for name, path in figures.items():
        parts.append(f"- **{name}**: `{path}`\n" if path
                     else f"- **{name}**: _not generated (experiment missing)_\n")

    text = "".join(parts)
    (RESULTS / "RESULTS.md").write_text(text)
    return text


def main() -> int:
    text = build_report()
    print(text[:4000])
    print("...")
    print(f"\nwritten to {RESULTS}/RESULTS.md and {FIGURES}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Build an interactive objective dashboard for the real-world seeded ABM.

    venv/bin/python run_objective_dashboard.py

Outputs:
  - objective_state_scores.csv
  - objective_dashboard.html

Open objective_dashboard.html in a browser. It has a step slider, objective
buttons, and a signed country graph whose nodes are countries and whose edges
show cooperative, hostile, and exception relationships.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib").resolve()))
Path(os.environ["MPLCONFIGDIR"]).mkdir(exist_ok=True)

import numpy as np
import pandas as pd

from geo_abm import GeoModel, metrics, rules


LEVEL_TARGETS = {
    "low": 0.15,
    "low-medium": 0.30,
    "medium": 0.50,
    "high": 0.75,
    "very high": 0.95,
    "neutral": 0.50,
    "negative": 0.05,
}

LEVEL_WEIGHTS = {
    "low": 0.35,
    "low-medium": 0.45,
    "medium": 0.60,
    "high": 0.80,
    "very high": 1.00,
    "neutral": 0.00,
    "negative": 0.90,
}

OBJECTIVES = {
    "Climate Mitigation": {
        "why": "Broad climate agreements need cross-bloc openness, stable alliances, and strong economic interaction.",
        "levels": {
            "homophily": "medium",
            "structural_balance": "high",
            "power_parity": "low",
            "gravity": "high",
            "global_cooperation": "very high",
        },
    },
    "Climate Adaptation": {
        "why": "Adaptation depends on regional cooperation, aid, disaster response, and practical economic links.",
        "levels": {
            "homophily": "medium",
            "structural_balance": "medium",
            "power_parity": "low",
            "gravity": "very high",
            "global_cooperation": "high",
        },
    },
    "Economic Growth": {
        "why": "Trade and capability-weighted interaction dominate, so ideology matters less than economic connectedness.",
        "levels": {
            "homophily": "low-medium",
            "structural_balance": "medium",
            "power_parity": "low",
            "gravity": "very high",
            "global_cooperation": "high",
        },
    },
    "Peace and Stability": {
        "why": "Stable alliance patterns reduce frustrated triads and recurring conflict incentives.",
        "levels": {
            "homophily": "medium",
            "structural_balance": "very high",
            "power_parity": "low",
            "gravity": "medium",
            "global_cooperation": "high",
        },
    },
}


def off_diagonal_values(matrix: np.ndarray) -> np.ndarray:
    n = matrix.shape[0]
    return matrix[np.triu_indices(n, k=1)]


def normalize_01(value: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    return float(np.clip((value - low) / (high - low), 0.0, 1.0))


def homophily_fit(ties: np.ndarray, alignment: np.ndarray) -> float:
    target = rules.homophily_target(alignment)
    diff = np.abs(off_diagonal_values(ties) - off_diagonal_values(target))
    return float(np.clip(1.0 - diff.mean() / 2.0, 0.0, 1.0))


def gravity_cooperation(ties: np.ndarray, power: np.ndarray) -> float:
    gravity = rules.gravity_weights(power)
    positive_ties = np.clip(ties, 0.0, None)
    denom = float(gravity.sum())
    if denom <= 1e-12:
        return 0.0
    return float(np.clip((positive_ties * gravity).sum() / denom, 0.0, 1.0))


def hostile_power_parity(ties: np.ndarray, power: np.ndarray) -> float:
    parity = rules.power_parity(power)
    rivalry = np.clip(-ties, 0.0, None)
    denom = float(rivalry.sum())
    if denom <= 1e-12:
        return 0.0
    return float(np.clip((parity * rivalry).sum() / denom, 0.0, 1.0))


def current_features(model: GeoModel) -> dict[str, float]:
    clean = metrics.camp_cleanliness(model.ties, model.relationship_threshold)
    exception_counts = metrics.camp_exception_counts(model.ties, model.relationship_threshold)
    return {
        "mean_cooperation": metrics.mean_cooperation(model.ties),
        "polarization": metrics.polarization(model.ties),
        "balanced_fraction": metrics.balanced_triad_fraction(model.ties),
        "n_camps": clean["n_camps"],
        "camp_cleanliness": clean["camp_cleanliness"],
        "camp_exception_rate": clean["camp_exception_rate"],
        "cross_camp_nonhostile_rate": exception_counts["cross_camp_nonhostile_rate"],
        "countries_with_cross_exceptions": exception_counts["countries_with_cross_exceptions"],
        "homophily": homophily_fit(model.ties, model.alignment),
        "structural_balance": metrics.balanced_triad_fraction(model.ties),
        "power_parity": hostile_power_parity(model.ties, model.power),
        "gravity": gravity_cooperation(model.ties, model.power),
        "global_cooperation": normalize_01(metrics.mean_cooperation(model.ties), -1.0, 1.0),
    }


def score_objective(features: dict[str, float], objective: dict) -> float:
    weighted_score = 0.0
    total_weight = 0.0
    for feature, level in objective["levels"].items():
        target = LEVEL_TARGETS[level]
        weight = LEVEL_WEIGHTS[level]
        if weight == 0.0:
            continue
        closeness = 1.0 - abs(features[feature] - target)
        weighted_score += weight * closeness
        total_weight += weight
    if total_weight <= 1e-12:
        return 0.0
    return float(np.clip(weighted_score / total_weight, 0.0, 1.0))


def simulate(steps: int, relationship_threshold: float, seed: int):
    model = GeoModel.from_real_data(
        cooperation_bias=-0.15,
        influence=0.0,
        relationship_threshold=relationship_threshold,
        seed=seed,
    )
    rows = []
    graph_steps = []
    node_order = list(model.names)

    for step in range(steps + 1):
        features = current_features(model)
        row = {"seed": seed, "step": step, **features}
        for objective_name, objective in OBJECTIVES.items():
            score_col = objective_name.lower().replace(" ", "_").replace("and", "and") + "_score"
            row[score_col] = score_objective(features, objective)
        rows.append(row)

        audit = metrics.relationship_audit(
            model.ties,
            threshold=relationship_threshold,
            names=model.names,
        )
        camps = metrics.ally_camps(model.ties, relationship_threshold)
        camp_by_node = {}
        for camp_idx, camp in enumerate(camps):
            for node in camp:
                camp_by_node[node] = camp_idx

        nodes = []
        for idx, name in enumerate(model.names):
            nodes.append(
                {
                    "id": idx,
                    "name": name,
                    "camp": int(camp_by_node[idx]),
                    "power": float(model.power[idx]),
                    "alignment": float(model.alignment[idx]),
                }
            )

        name_to_idx = {name: i for i, name in enumerate(model.names)}
        edges = []
        for edge in audit:
            tie = float(edge["tie"])
            edges.append(
                {
                    "source": name_to_idx[edge["country_i"]],
                    "target": name_to_idx[edge["country_j"]],
                    "tie": tie,
                    "sameCamp": bool(edge["same_camp"]),
                    "isException": bool(edge["is_exception"]),
                    "exceptionType": edge["exception_type"],
                    "expected": edge["expected_relationship"],
                }
            )

        graph_steps.append(
            {
                "step": step,
                "nodes": nodes,
                "edges": edges,
                "features": row,
            }
        )

        if step < steps:
            model.step()

    return pd.DataFrame(rows), graph_steps, node_order


def objective_summary(scores: pd.DataFrame) -> list[dict]:
    summaries = []
    for name, objective in OBJECTIVES.items():
        col = name.lower().replace(" ", "_").replace("and", "and") + "_score"
        idx = scores[col].idxmax()
        record = scores.loc[idx].to_dict()
        summaries.append(
            {
                "name": name,
                "scoreColumn": col,
                "bestStep": int(record["step"]),
                "bestScore": float(record[col]),
                "why": objective["why"],
                "levels": objective["levels"],
                "bestMetrics": {
                    "n_camps": float(record["n_camps"]),
                    "mean_cooperation": float(record["mean_cooperation"]),
                    "polarization": float(record["polarization"]),
                    "balanced_fraction": float(record["balanced_fraction"]),
                    "cross_camp_nonhostile_rate": (
                        None
                        if pd.isna(record["cross_camp_nonhostile_rate"])
                        else float(record["cross_camp_nonhostile_rate"])
                    ),
                    "camp_exception_rate": float(record["camp_exception_rate"]),
                    "gravity": float(record["gravity"]),
                    "power_parity": float(record["power_parity"]),
                    "global_cooperation": float(record["global_cooperation"]),
                },
            }
        )
    return summaries


def write_html(payload: dict, output_path: Path):
    payload_json = json.dumps(payload, separators=(",", ":"))
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Objective-Based Country Relationship Dynamics</title>
  <style>
    :root {{
      --bg: #f7f8fb;
      --ink: #1e2430;
      --muted: #5d6677;
      --line: #d8deea;
      --panel: #ffffff;
      --blue: #4c78a8;
      --green: #2ca02c;
      --red: #d62728;
      --orange: #f28e2b;
      --purple: #7b3294;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    header {{
      padding: 18px 24px 10px;
      border-bottom: 1px solid var(--line);
      background: var(--panel);
    }}
    h1 {{ margin: 0 0 4px; font-size: 22px; }}
    .sub {{ color: var(--muted); font-size: 13px; }}
    main {{
      display: grid;
      grid-template-columns: 330px 1fr;
      gap: 16px;
      padding: 16px;
    }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
    }}
    .objectives {{
      display: grid;
      gap: 8px;
      margin-top: 10px;
    }}
    button {{
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 9px 10px;
      background: #fff;
      color: var(--ink);
      text-align: left;
      cursor: pointer;
      font: inherit;
    }}
    button.active {{
      border-color: var(--blue);
      background: #eaf2ff;
    }}
    .controls {{
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 10px;
      align-items: center;
      margin-bottom: 12px;
    }}
    input[type="range"] {{ width: 100%; }}
    select {{
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 7px 9px;
      background: white;
      font: inherit;
    }}
    .stats {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 8px;
      margin: 12px 0;
    }}
    .stat {{
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px;
      min-height: 62px;
    }}
    .label {{ color: var(--muted); font-size: 12px; }}
    .value {{ font-size: 19px; font-weight: 650; margin-top: 3px; }}
    svg {{
      width: 100%;
      height: 680px;
      display: block;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fbfcff;
    }}
    .edge {{ fill: none; stroke-linecap: round; }}
    .edge.hostile {{ stroke: var(--red); stroke-dasharray: 7 5; }}
    .edge.cooperative {{ stroke: var(--green); }}
    .edge.cross-exception {{ stroke: var(--orange); }}
    .edge.cooperative-exception {{ stroke: #009e73; }}
    .edge.within-exception {{ stroke: var(--purple); stroke-dasharray: 3 4; }}
    .node-label {{ font-size: 11px; fill: #1f2937; pointer-events: none; }}
    .camp-label {{ font-size: 16px; font-weight: 700; fill: #374151; }}
    .note {{ color: var(--muted); line-height: 1.35; font-size: 13px; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 12px; }}
    td {{ border-top: 1px solid var(--line); padding: 5px 2px; }}
    td:last-child {{ text-align: right; font-weight: 600; }}
    .legend {{ display: flex; flex-wrap: wrap; gap: 12px; font-size: 12px; margin-top: 8px; color: var(--muted); }}
    .swatch {{ display: inline-block; width: 26px; height: 3px; vertical-align: middle; margin-right: 5px; }}
    @media (max-width: 900px) {{
      main {{ grid-template-columns: 1fr; }}
      svg {{ height: 620px; }}
      .stats {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>Objective-Based Country Relationship Dynamics</h1>
    <div class="sub">Real-world seeded ABM. Choose a goal to jump to the timestep whose rule profile is closest to that objective.</div>
  </header>
  <main>
    <aside class="panel">
      <div class="label">Objective buttons</div>
      <div id="objectiveButtons" class="objectives"></div>
      <table id="objectiveTable"></table>
      <p id="objectiveWhy" class="note"></p>
    </aside>
    <section class="panel">
      <div class="controls">
        <label>
          <span class="label">Step: <strong id="stepLabel">0</strong></span>
          <input id="stepSlider" type="range" min="0" max="{payload["maxStep"]}" value="0">
        </label>
        <label>
          <span class="label">Edges</span><br>
          <select id="edgeMode">
            <option value="exceptions">Exceptions only</option>
            <option value="hostile">Hostile ties only</option>
            <option value="exceptions_hostile">Exceptions + hostile ties</option>
            <option value="important">Exceptions + strong ties</option>
            <option value="all">All ties</option>
          </select>
        </label>
      </div>
      <div class="stats">
        <div class="stat"><div class="label">Selected objective score</div><div id="scoreValue" class="value">-</div></div>
        <div class="stat"><div class="label">Camps</div><div id="campsValue" class="value">-</div></div>
        <div class="stat"><div class="label">Mean cooperation</div><div id="coopValue" class="value">-</div></div>
        <div class="stat"><div class="label">Cross-camp non-hostile rate</div><div id="messyValue" class="value">-</div></div>
      </div>
      <svg id="graph" viewBox="0 0 1000 680" role="img"></svg>
      <div class="legend">
        <span><span class="swatch" style="background:var(--green)"></span>Cooperative</span>
        <span><span class="swatch" style="background:var(--red)"></span>Hostile</span>
        <span><span class="swatch" style="background:var(--orange)"></span>Cross-bloc not hostile exception</span>
        <span><span class="swatch" style="background:var(--purple)"></span>Within-bloc exception</span>
      </div>
    </section>
  </main>
  <script>
    const DATA = {payload_json};
    const svg = document.getElementById("graph");
    const slider = document.getElementById("stepSlider");
    const edgeMode = document.getElementById("edgeMode");
    const objectiveButtons = document.getElementById("objectiveButtons");
    const objectiveTable = document.getElementById("objectiveTable");
    const objectiveWhy = document.getElementById("objectiveWhy");
    let selectedObjective = DATA.objectives[0];

    function fmt(value, digits = 2) {{
      if (value === null || value === undefined || Number.isNaN(value)) return "n/a";
      return Number(value).toFixed(digits);
    }}

    function scoreColumn(objective) {{
      return objective.scoreColumn;
    }}

    function stepData(step) {{
      return DATA.steps.find(s => s.step === Number(step)) || DATA.steps[0];
    }}

    function nodePositions(nodes) {{
      const camps = [...new Set(nodes.map(n => n.camp))].sort((a, b) => a - b);
      const positions = new Map();
      camps.forEach((camp, campIndex) => {{
        const campNodes = nodes.filter(n => n.camp === camp).sort((a, b) => a.name.localeCompare(b.name));
        const x = camps.length === 1 ? 500 : 180 + campIndex * (640 / Math.max(1, camps.length - 1));
        campNodes.forEach((node, i) => {{
          const y = 80 + i * (520 / Math.max(1, campNodes.length - 1));
          positions.set(node.id, {{ x, y }});
        }});
      }});
      return positions;
    }}

    function edgeClass(edge) {{
      if (edge.exceptionType === "cross_camp_not_hostile" && edge.tie > DATA.relationshipThreshold) return "cooperative-exception";
      if (edge.exceptionType === "cross_camp_not_hostile") return "cross-exception";
      if (edge.exceptionType === "within_camp_not_cooperative") return "within-exception";
      if (edge.tie > 0) return "cooperative";
      return "hostile";
    }}

    function keepEdge(edge) {{
      if (edgeMode.value === "exceptions") return edge.isException;
      if (edgeMode.value === "hostile") return edge.tie < -DATA.relationshipThreshold;
      if (edgeMode.value === "exceptions_hostile") return edge.isException || edge.tie < -DATA.relationshipThreshold;
      if (edgeMode.value === "important") return edge.isException || Math.abs(edge.tie) >= 0.35;
      return true;
    }}

    function edgeOpacity(edge) {{
      if (edge.isException) return 0.62;
      if (edge.tie < -DATA.relationshipThreshold) return 0.74;
      if (edge.tie > DATA.relationshipThreshold) return 0.34;
      return 0.16;
    }}

    function edgeWidth(edge) {{
      const base = edge.tie < -DATA.relationshipThreshold ? 1.8 : 1.0;
      return base + Math.max(Math.abs(edge.tie), 0.08) * 3.2;
    }}

    function draw(step) {{
      const s = stepData(step);
      const positions = nodePositions(s.nodes);
      svg.innerHTML = "";

      const camps = [...new Set(s.nodes.map(n => n.camp))].sort((a, b) => a - b);
      camps.forEach(camp => {{
        const campNodes = s.nodes.filter(n => n.camp === camp);
        const xs = campNodes.map(n => positions.get(n.id).x);
        const x = xs.reduce((a, b) => a + b, 0) / xs.length;
        const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
        text.setAttribute("x", x);
        text.setAttribute("y", 42);
        text.setAttribute("text-anchor", "middle");
        text.setAttribute("class", "camp-label");
        text.textContent = `Bloc ${{camp + 1}}`;
        svg.appendChild(text);
      }});

      s.edges
        .filter(keepEdge)
        .sort((a, b) => {{
          const rank = edge => edge.isException ? 2 : edge.tie < -DATA.relationshipThreshold ? 1 : 0;
          return rank(a) - rank(b);
        }})
        .forEach(edge => {{
        const a = positions.get(edge.source);
        const b = positions.get(edge.target);
        const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
        line.setAttribute("x1", a.x);
        line.setAttribute("y1", a.y);
        line.setAttribute("x2", b.x);
        line.setAttribute("y2", b.y);
        line.setAttribute("class", `edge ${{edgeClass(edge)}}`);
        line.setAttribute("stroke-width", edgeWidth(edge));
        line.setAttribute("opacity", edgeOpacity(edge));
        svg.appendChild(line);
      }});

      s.nodes.forEach(node => {{
        const p = positions.get(node.id);
        const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
        circle.setAttribute("cx", p.x);
        circle.setAttribute("cy", p.y);
        circle.setAttribute("r", 8 + Math.sqrt(node.power) * 2.4);
        circle.setAttribute("fill", DATA.campColors[node.camp % DATA.campColors.length]);
        circle.setAttribute("stroke", "white");
        circle.setAttribute("stroke-width", "1.5");
        svg.appendChild(circle);

        const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
        label.setAttribute("x", p.x + 13);
        label.setAttribute("y", p.y + 4);
        label.setAttribute("class", "node-label");
        label.textContent = node.name;
        svg.appendChild(label);
      }});

      document.getElementById("stepLabel").textContent = s.step;
      document.getElementById("scoreValue").textContent = fmt(s.features[scoreColumn(selectedObjective)], 3);
      document.getElementById("campsValue").textContent = fmt(s.features.n_camps, 0);
      document.getElementById("coopValue").textContent = fmt(s.features.mean_cooperation, 2);
      document.getElementById("messyValue").textContent = fmt(s.features.cross_camp_nonhostile_rate, 2);
    }}

    function renderObjective(objective) {{
      selectedObjective = objective;
      [...objectiveButtons.children].forEach(button => {{
        button.classList.toggle("active", button.dataset.name === objective.name);
      }});
      objectiveTable.innerHTML = Object.entries(objective.levels).map(([key, value]) =>
        `<tr><td>${{key.replaceAll("_", " ")}}</td><td>${{value}}</td></tr>`
      ).join("");
      objectiveWhy.textContent = `${{objective.why}} Best step: ${{objective.bestStep}}. Best score: ${{fmt(objective.bestScore, 3)}}.`;
      slider.value = objective.bestStep;
      draw(objective.bestStep);
    }}

    DATA.objectives.forEach(objective => {{
      const button = document.createElement("button");
      button.textContent = `${{objective.name}} - best step ${{objective.bestStep}}`;
      button.dataset.name = objective.name;
      button.addEventListener("click", () => renderObjective(objective));
      objectiveButtons.appendChild(button);
    }});

    slider.addEventListener("input", () => draw(slider.value));
    edgeMode.addEventListener("change", () => draw(slider.value));
    renderObjective(selectedObjective);
  </script>
</body>
</html>
"""
    output_path.write_text(html)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--relationship-threshold", type=float, default=0.1)
    args = parser.parse_args()

    scores, graph_steps, node_order = simulate(args.steps, args.relationship_threshold, args.seed)
    scores.to_csv("objective_state_scores.csv", index=False)

    payload = {
        "maxStep": args.steps,
        "seed": args.seed,
        "relationshipThreshold": args.relationship_threshold,
        "nodeOrder": node_order,
        "campColors": ["#5B8FF9", "#61DDAA", "#F6BD16", "#E8684A", "#6DC8EC"],
        "objectives": objective_summary(scores),
        "steps": graph_steps,
    }
    write_html(payload, Path("objective_dashboard.html"))

    print("Wrote objective_state_scores.csv")
    print("Wrote objective_dashboard.html")
    print("\nBest steps by objective:")
    for summary in payload["objectives"]:
        print(f"  {summary['name']}: step {summary['bestStep']} (score={summary['bestScore']:.3f})")


if __name__ == "__main__":
    main()

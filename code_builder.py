"""
code_builder.py — Module 4
Deterministic compiler: internal graph JSON → OR-Tools CP-SAT Python.
No LLM involvement. Output is always syntactically valid.
Follows canonical 9-section template.

Depends on: instance_parser.py (schema only)
"""
from typing import Optional


def build_code(graph: dict) -> str:
    """
    Main entry point. Dispatches on constraint type.
    Returns a complete, executable Python code string.
    """
    ctype = graph['meta']['type']
    if ctype == 'TYPE-A':
        return _build_type_a(graph)
    elif ctype == 'TYPE-B':
        return _build_type_b(graph)
    elif ctype == 'TYPE-C':
        # TYPE-C = TYPE-A + generalized lags (lag_max)
        # Same code structure as TYPE-A — CodeBuilder handles
        # lag_max via the shared precedence compiler
        return _build_type_a(graph)
    else:
        raise ValueError(f"Unknown constraint type: {ctype}")


# ══════════════════════════════════════════════════════════════════
# SHARED SECTIONS (identical across all types)
# ══════════════════════════════════════════════════════════════════

def _section_1_imports() -> str:
    return (
        "from ortools.sat.python import cp_model\n"
        "import json, sys\n"
    )


def _section_2_data(graph: dict) -> str:
    """Embeds instance data directly into the code as Python dicts."""
    lines = []

    # Activities
    ctype = graph['meta']['type']
    if ctype in ('TYPE-A', 'TYPE-C'):
        acts = []
        for a in graph['activities']:
            acts.append(
                f"  {a['id']}: {{'dur': {a['duration']}, "
                f"'dem': {dict(a['demands'])}}}")
        lines.append("activities = {\n" + ",\n".join(acts) + "\n}")
    elif ctype == 'TYPE-B':
        acts = []
        for a in graph['activities']:
            modes_str = []
            for m in a['modes']:
                modes_str.append(
                    f"    {{'mid': {m['mode_id']}, "
                    f"'dur': {m['duration']}, "
                    f"'rdem': {dict(m['renewable_demands'])}, "
                    f"'ndem': {dict(m['nonrenewable_demands'])}}}")
            acts.append(
                f"  {a['id']}: [\n" + ",\n".join(modes_str) + "\n  ]")
        lines.append("activities = {\n" + ",\n".join(acts) + "\n}")

        # Resources
    res_lines = []
    for r in graph['resources']:
        entry = f"  '{r['id']}': {{'cap': {r['capacity']}"
        if r.get('type') == 'nonrenewable':
            entry += ", 'type': 'nr'"
        else:
            dt_list = r.get('downtime', [])
            if dt_list:
                entry += f", 'downtime': {dt_list}"
            else:
                entry += ", 'downtime': []"
        entry += "}"
        res_lines.append(entry)
    lines.append("resources = {\n" + ",\n".join(res_lines) + "\n}")

    # Precedences
    prec_entries = []
    for p in graph['precedences']:
        lag_max_str = str(p['lag_max']) if p.get('lag_max') is not None else "None"
        prec_entries.append(
            f"  {{'i': {p['i']}, 'j': {p['j']}, "
            f"'rel': '{p['relation']}', "
            f"'lmin': {p['lag_min']}, 'lmax': {lag_max_str}}}")
    lines.append("precedences = [\n" + ",\n".join(prec_entries) + "\n]")

    # Objective
    obj = graph['objective']
    obj_str = f"objective = {{'type': '{obj['type']}'"
    if obj.get('task_weights'):
        obj_str += f", 'weights': {obj['task_weights']}"
    if obj.get('task_deadlines'):
        obj_str += f", 'deadlines': {obj['task_deadlines']}"
    obj_str += "}"
    lines.append(obj_str)

    return "\n".join(lines)


def _section_3_model_setup(graph: dict) -> str:
    """Compute safe horizon and create model."""
    # Compute horizon at code-generation time and embed as constant
    dur_sum = 0
    for a in graph['activities']:
        if 'duration' in a:
            dur_sum += a['duration']
        elif 'modes' in a:
            dur_sum += max(m['duration'] for m in a['modes'])

    lag_sum = sum(
        p['lag_min'] for p in graph['precedences']
        if p.get('lag_min', 0) > 0)

    dt_sum = sum(
        dt['end'] - dt['start']
        for r in graph['resources']
        for dt in r.get('downtime', []))

    horizon = dur_sum + lag_sum + dt_sum

    return (
        "model = cp_model.CpModel()\n"
        f"horizon = {horizon}\n"
    )


def _section_6_precedences() -> str:
    """Precedence constraint compiler — handles all 4 PDM types + lag_max."""
    return (
        "# ── Precedence constraints ──\n"
        "for p in precedences:\n"
        "    i, j = p['i'], p['j']\n"
        "    rel  = p['rel']\n"
        "    lmin = p['lmin']\n"
        "    lmax = p['lmax']\n"
        "\n"
        "    # Minimum lag (forward constraint)\n"
        "    if   rel == 'FS': model.Add(starts[j] >= ends[i]   + lmin)\n"
        "    elif rel == 'SS': model.Add(starts[j] >= starts[i] + lmin)\n"
        "    elif rel == 'FF': model.Add(ends[j]   >= ends[i]   + lmin)\n"
        "    elif rel == 'SF': model.Add(ends[j]   >= starts[i] + lmin)\n"
        "\n"
        "    # Maximum lag (backward constraint) — only if lag_max is set\n"
        "    if lmax is not None:\n"
        "        if   rel == 'FS': model.Add(starts[j] <= ends[i]   + lmax)\n"
        "        elif rel == 'SS': model.Add(starts[j] <= starts[i] + lmax)\n"
        "        elif rel == 'FF': model.Add(ends[j]   <= ends[i]   + lmax)\n"
        "        elif rel == 'SF': model.Add(ends[j]   <= starts[i] + lmax)\n"
    )


def _section_7_objective() -> str:
    """Objective compiler — handles makespan and weighted tardiness."""
    return (
        "# ── Objective ──\n"
        "if objective['type'] == 'minimize_makespan':\n"
        "    makespan = model.NewIntVar(0, horizon, 'makespan')\n"
        "    model.AddMaxEquality(makespan, list(ends.values()))\n"
        "    model.Minimize(makespan)\n"
        "\n"
        "elif objective['type'] == 'minimize_weighted_tardiness':\n"
        "    tard_terms = []\n"
        "    for aid in activities:\n"
        "        dl = objective['deadlines'][str(aid)]\n"
        "        wt = objective['weights'][str(aid)]\n"
        "        tard = model.NewIntVar(0, horizon, f'tard{aid}')\n"
        "        model.Add(tard >= ends[aid] - dl)\n"
        "        tard_terms.append(wt * tard)\n"
        "    model.Minimize(sum(tard_terms))\n"
    )


def _section_8_solve() -> str:
    return (
        "# ── Solve ──\n"
        "solver = cp_model.CpSolver()\n"
        "solver.parameters.max_time_in_seconds = 60.0\n"
        "status = solver.Solve(model)\n"
    )


def _section_9_output() -> str:
    return (
        "# ── Output ──\n"
        "if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):\n"
        "    result = {\n"
        "        'status': 'optimal' if status == cp_model.OPTIMAL else 'feasible',\n"
        "        'objective': solver.ObjectiveValue(),\n"
        "        'schedule': {str(aid): solver.Value(starts[aid]) for aid in starts}\n"
        "    }\n"
        "    print(json.dumps(result))\n"
        "else:\n"
        "    print(json.dumps({'status': 'infeasible'}))\n"
    )


# ══════════════════════════════════════════════════════════════════
# TYPE-A SPECIFIC SECTIONS
# ══════════════════════════════════════════════════════════════════

def _section_4a_variables() -> str:
    """TYPE-A: one fixed-size interval per activity."""
    return (
        "# ── Decision variables (TYPE-A) ──\n"
        "starts    = {}\n"
        "ends      = {}\n"
        "intervals = {}\n"
        "for aid, a in activities.items():\n"
        "    s = model.NewIntVar(0, horizon, f's{aid}')\n"
        "    e = model.NewIntVar(0, horizon, f'e{aid}')\n"
        "    iv = model.NewFixedSizeIntervalVar(s, a['dur'], e, f'iv{aid}')\n"
        "    starts[aid] = s\n"
        "    ends[aid]   = e\n"
        "    intervals[aid] = iv\n"
    )


def _section_5a_resources() -> str:
    """TYPE-A: cumulative constraints with downtime dummy tasks."""
    return (
        "# ── Resource constraints (TYPE-A) ──\n"
        "for rname, r in resources.items():\n"
        "    if r.get('type') == 'nr':\n"
        "        continue  # nonrenewable handled separately\n"
        "    r_ivs     = []\n"
        "    r_demands = []\n"
        "    for aid, a in activities.items():\n"
        "        d = a['dem'].get(rname, 0)\n"
        "        if d > 0:\n"
        "            r_ivs.append(intervals[aid])\n"
        "            r_demands.append(d)\n"
        "\n"
        "    # Downtime dummy tasks — ALL windows\n"
        "    for dt in r.get('downtime', []):\n"
        "        dt_s  = model.NewConstant(dt['start'])\n"
        "        dt_e  = model.NewConstant(dt['end'])\n"
        "        dt_iv = model.NewFixedSizeIntervalVar(\n"
        "            dt_s, dt['end'] - dt['start'],\n"
        "            dt_e, f'dt_{rname}_{dt[\"start\"]}')\n"
        "        r_ivs.append(dt_iv)\n"
        "        r_demands.append(dt['delta'])\n"
        "\n"
        "    if r_ivs:\n"
        "        model.AddCumulative(r_ivs, r_demands, r['cap'])\n"
    )


# ══════════════════════════════════════════════════════════════════
# TYPE-B SPECIFIC SECTIONS
# ══════════════════════════════════════════════════════════════════

def _section_4b_variables() -> str:
    """TYPE-B: BoolVar mode selection + optional intervals."""
    return (
        "# ── Decision variables (TYPE-B) ──\n"
        "starts     = {}\n"
        "ends       = {}\n"
        "mode_bools = {}  # (aid, mode_id) → BoolVar\n"
        "mode_ivs   = {}  # (aid, mode_id) → OptionalIntervalVar\n"
        "\n"
        "for aid, modes in activities.items():\n"
        "    s = model.NewIntVar(0, horizon, f's{aid}')\n"
        "    e = model.NewIntVar(0, horizon, f'e{aid}')\n"
        "    starts[aid] = s\n"
        "    ends[aid]   = e\n"
        "\n"
        "    bools_for_activity = []\n"
        "    for m in modes:\n"
        "        mid = m['mid']\n"
        "        b = model.NewBoolVar(f'b_{aid}_{mid}')\n"
        "        mode_bools[(aid, mid)] = b\n"
        "        bools_for_activity.append(b)\n"
        "\n"
        "        iv = model.NewOptionalFixedSizeIntervalVar(\n"
        "            s, m['dur'], e, b, f'iv_{aid}_{mid}')\n"
        "        mode_ivs[(aid, mid)] = iv\n"
        "\n"
        "    # Exactly one mode selected per activity\n"
        "    model.AddExactlyOne(bools_for_activity)\n"
    )


def _section_5b_resources() -> str:
    """TYPE-B: cumulative with optional intervals + NR budget."""
    return (
        "# ── Resource constraints (TYPE-B) ──\n"
        "for rname, r in resources.items():\n"
        "    if r.get('type') == 'nr':\n"
        "        # Nonrenewable: total budget constraint\n"
        "        nr_terms = []\n"
        "        for aid, modes in activities.items():\n"
        "            for m in modes:\n"
        "                mid = m['mid']\n"
        "                d   = m['ndem'].get(rname, 0)\n"
        "                if d > 0:\n"
        "                    nr_terms.append(d * mode_bools[(aid, mid)])\n"
        "        if nr_terms:\n"
        "            model.Add(sum(nr_terms) <= r['cap'])\n"
        "        continue\n"
        "\n"
        "    # Renewable: cumulative with optional intervals\n"
        "    r_ivs     = []\n"
        "    r_demands = []\n"
        "    for aid, modes in activities.items():\n"
        "        for m in modes:\n"
        "            mid = m['mid']\n"
        "            d   = m['rdem'].get(rname, 0)\n"
        "            if d > 0:\n"
        "                r_ivs.append(mode_ivs[(aid, mid)])\n"
        "                r_demands.append(d)\n"
        "\n"
        "    # Downtime dummy tasks\n"
        "    for dt in r.get('downtime', []):\n"
        "        dt_s  = model.NewConstant(dt['start'])\n"
        "        dt_e  = model.NewConstant(dt['end'])\n"
        "        dt_iv = model.NewFixedSizeIntervalVar(\n"
        "            dt_s, dt['end'] - dt['start'],\n"
        "            dt_e, f'dt_{rname}_{dt[\"start\"]}')\n"
        "        r_ivs.append(dt_iv)\n"
        "        r_demands.append(dt['delta'])\n"
        "\n"
        "    if r_ivs:\n"
        "        model.AddCumulative(r_ivs, r_demands, r['cap'])\n"
    )


# ══════════════════════════════════════════════════════════════════
# ASSEMBLY
# ══════════════════════════════════════════════════════════════════
def _build_type_a(graph: dict) -> str:
    """Assembles canonical TYPE-A (and TYPE-C) code."""
    sections = [
        _section_1_imports(),
        _section_2_data(graph),
        _section_3_model_setup(graph),
        _section_4a_variables(),
        _section_5a_resources(),
        _section_6_precedences(),
        _section_7_objective(),
        _section_8_solve(),
        _section_9_output(),
    ]
    return "\n".join(sections)


def _build_type_b(graph: dict) -> str:
    """Assembles canonical TYPE-B code."""
    sections = [
        _section_1_imports(),
        _section_2_data(graph),
        _section_3_model_setup(graph),
        _section_4b_variables(),
        _section_5b_resources(),
        _section_6_precedences(),
        _section_7_objective(),
        _section_8_solve(),
        _section_9_output(),
    ]
    return "\n".join(sections)


# ══════════════════════════════════════════════════════════════════
# TARGET m SERIALIZER (separate from code — schema boundary)
# ══════════════════════════════════════════════════════════════════

def serialize_target_m(graph: dict) -> dict:
    """
    Strips backend fields from internal graph JSON.
    Returns target m: only fields inferrable from p alone.
    This is what the LLM learns to generate.
    """
    ctype = graph['meta']['type']

    # Activities — branch on type
    if ctype in ('TYPE-A', 'TYPE-C'):
        activities = [
            {"id": a['id'],
             "duration": a['duration'],
             "demands": dict(a['demands'])}
            for a in graph['activities']
        ]
    elif ctype == 'TYPE-B':
        activities = []
        for a in graph['activities']:
            modes = [
                {"mode_id": m['mode_id'],
                 "duration": m['duration'],
                 "renewable_demands": dict(m['renewable_demands']),
                 "nonrenewable_demands": dict(m['nonrenewable_demands'])}
                for m in a['modes']
            ]
            activities.append({"id": a['id'], "modes": modes})

    # Resources
    resources = []
    for r in graph['resources']:
        entry = {"id": r['id'], "capacity": r['capacity']}
        if r.get('type') == 'nonrenewable':
            entry['type'] = 'nonrenewable'
        else:
            entry['downtime'] = r.get('downtime', [])
        resources.append(entry)

    # Precedences
    precedences = [
        {"i": p['i'], "j": p['j'],
         "relation": p['relation'],
         "lag_min": p['lag_min'],
         "lag_max": p.get('lag_max')}
        for p in graph['precedences']
    ]

    # Objective
    objective = {
        "type": graph['objective']['type'],
        "task_weights": graph['objective'].get('task_weights'),
        "task_deadlines": graph['objective'].get('task_deadlines')
    }

    return {
        "constraint_type": ctype,
        "activities": activities,
        "resources": resources,
        "precedences": precedences,
        "objective": objective
        # NO: structural_hash, split, instance_id, mutations_applied,
        #     RF, RS, source — these are backend-only fields
    }
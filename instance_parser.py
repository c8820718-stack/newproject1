"""
instance_parser.py — Module 1
Parses PSPLIB .sm and MMLIB .mm files into internal graph JSON.
Computes: RF, RS, critical_path_length, structural_hash.
No dependencies on other pipeline modules.
"""
import re, json, hashlib, copy
from pathlib import Path
from typing import Optional


# ══════════════════════════════════════════════════════════════════
# PSPLIB .sm parser (TYPE-A)
# ══════════════════════════════════════════════════════════════════

def parse_psplib(filepath: str | Path,
                 instance_id: str,
                 split: str = "train") -> dict:
    """
    Parses a PSPLIB .sm file.
    Returns internal graph JSON (schema locked in Week 2).

    PSPLIB .sm file structure:
    ──────────────────────────
    file with basedata            :
    initial value random generator: ...
    projects                     :  1
    jobs (incl. supersource/sink) : 32  ← n_activities + 2
    horizon                      : 138
    RESOURCES
    - renewable                 :  4   R   ← n_resources
    - nonrenewable              :  0   N
    - doubly constrained        :  0   D
    PROJECT INFORMATION:
    ...
    PRECEDENCE RELATIONS:
    jobnr.    #modes  #successors   successors
       1        1          3           2   3   4
    ...
    REQUESTS/DURATIONS:
    jobnr. mode  duration  R 1  R 2  R 3  R 4
       1      1     0       0    0    0    0
    ...
    RESOURCEAVAILABILITIES:
      R 1  R 2  R 3  R 4
        4    2    3    2
    """
    lines = Path(filepath).read_text().splitlines()
    lines = [l.strip() for l in lines]

    # ── Parse header ──────────────────────────────────────────────
    n_jobs = _extract_int(lines, 'jobs')
    n_resources = _extract_int(lines, 'renewable')
    horizon = _extract_int(lines, 'horizon')

    # ── Parse precedence relations ────────────────────────────────
    prec_start = _find_section(lines, 'PRECEDENCE RELATIONS')
    prec_lines = _read_data_lines(lines, prec_start + 1)  # 偏移改为 +1

    precedences = []
    for line in prec_lines:
        parts = line.split()
        if len(parts) < 3:
            continue
        job_id = int(parts[0])
        n_succ = int(parts[2])
        successors = [int(parts[3 + k]) for k in range(n_succ)]
        for succ in successors:
            precedences.append({
                "i": job_id,
                "j": succ,
                "relation": "FS",
                "lag_min": 0,
                "lag_max": None
            })

    # ── Parse durations and resource demands ──────────────────────
    req_start = _find_section(lines, 'REQUESTS/DURATIONS')
    req_lines = _read_data_lines(lines, req_start + 1)  # 偏移改为 +1

    activities = []
    for line in req_lines:
        parts = line.split()
        if len(parts) < 3:
            continue
        job_id = int(parts[0])
        duration = int(parts[2])
        demands = {f"R{k + 1}": int(parts[3 + k])
                   for k in range(n_resources)}
        activities.append({
            "id": job_id,
            "duration": duration,
            "demands": demands
        })

    # ── Parse resource capacities ─────────────────────────────────
    res_start = _find_section(lines, 'RESOURCEAVAILABILITIES')
    res_lines = _read_data_lines(lines, res_start + 1)  # 使用安全函数读取
    cap_line = res_lines[0].split()  # 取第一行数据
    resources = [
        {"id": f"R{k + 1}", "capacity": int(cap_line[k]),
         "type": "renewable", "downtime": []}
        for k in range(n_resources)
    ]

    # ── Remove supersource (job 1) and supersink (job n_jobs) ─────
    # PSPLIB convention: job 1 = source (dur=0), job n_jobs = sink (dur=0)
    sink_id = n_jobs
    activities = [a for a in activities
                  if a['id'] != 1 and a['id'] != sink_id]
    precedences = [p for p in precedences
                   if p['i'] != 1 and p['j'] != sink_id
                   and p['i'] != sink_id and p['j'] != 1]

    # ── Compute metrics ───────────────────────────────────────────
    rf = _compute_RF(activities, resources)
    rs = _compute_RS(activities, resources, horizon)
    cp_length = _compute_critical_path(activities, precedences)
    net_density = _compute_network_density(activities, precedences)
    struct_hash = _compute_structural_hash(activities, precedences)

    return {
        "meta": {
            "instance_id": instance_id,
            "source": "PSPLIB",
            "type": "TYPE-A",
            "difficulty": None,  # set by stratifier
            "question_type": None,  # set by stratifier
            "RF": rf,
            "RS": rs,
            "split": split,
            "structural_hash": struct_hash
        },
        "activities": activities,
        "resources": resources,
        "precedences": precedences,
        "objective": {
            "type": "minimize_makespan",
            "task_weights": None,
            "task_deadlines": None
        },
        "mutations_applied": []
    }


# ══════════════════════════════════════════════════════════════════
# MMLIB .mm parser (TYPE-B)
# ══════════════════════════════════════════════════════════════════

def parse_mmlib(filepath: str | Path,
                instance_id: str,
                split: str = "train") -> dict:
    """
    Parses an MMLIB .mm file.
    MMLIB format adds a modes block per activity:
    jobnr.  #modes  #successors  successors
    jobnr.  mode  duration  R1  R2  NR1  NR2

    MMLIB resource structure:
    - renewable resources (R): capacity per period
    - nonrenewable resources (N): total budget for project
    """
    lines = Path(filepath).read_text().splitlines()
    lines = [l.strip() for l in lines]

    n_jobs = _extract_int(lines, 'jobs')
    n_renewable = _extract_int(lines, 'renewable')
    n_nonrenewable = _extract_int(lines, 'nonrenewable')
    n_resources = n_renewable + n_nonrenewable

    # ── Parse precedences ─────────────────
    prec_start = _find_section(lines, 'PRECEDENCE RELATIONS')
    prec_lines = _read_data_lines(lines, prec_start + 1)  # 改为 +1
    precedences = []
    for line in prec_lines:
        parts = line.split()
        if len(parts) < 3:
            continue
        job_id = int(parts[0])
        n_succ = int(parts[2])
        successors = [int(parts[3 + k]) for k in range(n_succ)]
        for succ in successors:
            precedences.append({
                "i": job_id, "j": succ,
                "relation": "FS", "lag_min": 0, "lag_max": None
            })

    # ── Parse multi-mode durations and demands ────────────────────
    req_start = _find_section(lines, 'REQUESTS/DURATIONS')
    req_lines = _read_data_lines(lines, req_start + 1)  # 改为 +1

    # Group lines by job number
    activities_dict = {}
    for line in req_lines:
        parts = line.split()
        if len(parts) < 3:
            continue
        job_id = int(parts[0])
        mode_id = int(parts[1])
        dur = int(parts[2])
        r_demands = {f"R{k + 1}": int(parts[3 + k])
                     for k in range(n_renewable)}
        nr_demands = {f"NR{k + 1}": int(parts[3 + n_renewable + k])
                      for k in range(n_nonrenewable)}
        if job_id not in activities_dict:
            activities_dict[job_id] = {"id": job_id, "modes": []}
        activities_dict[job_id]["modes"].append({
            "mode_id": mode_id,
            "duration": dur,
            "renewable_demands": r_demands,
            "nonrenewable_demands": nr_demands
        })
    activities = list(activities_dict.values())

    # ── Parse resource capacities ─────────────────────────────────
    res_start = _find_section(lines, 'RESOURCEAVAILABILITIES')
    res_lines = _read_data_lines(lines, res_start + 1)  # 使用安全函数读取
    cap_line = res_lines[0].split()  # 取第一行数据
    resources = (
            [{"id": f"R{k + 1}", "capacity": int(cap_line[k]),
              "type": "renewable", "downtime": []}
             for k in range(n_renewable)] +
            [{"id": f"NR{k + 1}", "capacity": int(cap_line[n_renewable + k]),
              "type": "nonrenewable"}
             for k in range(n_nonrenewable)]
    )

    # ── Remove supersource and supersink ──────────────────────────
    sink_id = n_jobs
    activities = [a for a in activities
                  if a['id'] != 1 and a['id'] != sink_id]
    precedences = [p for p in precedences
                   if p['i'] != 1 and p['j'] != sink_id
                   and p['i'] != sink_id and p['j'] != 1]

    rf = _compute_RF_multimode(activities, resources)
    rs = _compute_RS_multimode(activities, resources)
    cp_length = _compute_critical_path_multimode(activities, precedences)
    net_density = _compute_network_density(activities, precedences)
    struct_hash = _compute_structural_hash_multimode(activities, precedences)

    return {
        "meta": {
            "instance_id": instance_id,
            "source": "MMLIB",
            "type": "TYPE-B",
            "difficulty": None,
            "question_type": None,
            "RF": rf,
            "RS": rs,
            "split": split,
            "structural_hash": struct_hash
        },
        "activities": activities,
        "resources": resources,
        "precedences": precedences,
        "objective": {
            "type": "minimize_makespan",
            "task_weights": None,
            "task_deadlines": None
        },
        "mutations_applied": []
    }

    # ══════════════════════════════════════════════════════════════════


# Metric computations
# ══════════════════════════════════════════════════════════════════

def _compute_RF(activities: list, resources: list) -> float:
    """
    Resource Factor: average fraction of resource types used per activity.
    RF = (1 / (n * K)) * sum_{j,k} 1[d_jk > 0]
    where n = #activities, K = #resources.
    Range [0,1]. High RF → many resource types used → harder.
    Reference: Kolisch & Sprecher 1997.
    """
    n = len(activities)
    K = len(resources)
    if n == 0 or K == 0:
        return 0.0
    count = sum(
        1 for a in activities
        for r in resources
        if a['demands'].get(r['id'], 0) > 0
    )
    return count / (n * K)


def _compute_RF_multimode(activities: list, resources: list) -> float:
    """RF for multi-mode: use minimum-demand mode per activity."""
    n = len(activities)
    K = sum(1 for r in resources if r['type'] == 'renewable')
    if n == 0 or K == 0:
        return 0.0
    renewable_ids = [r['id'] for r in resources
                     if r['type'] == 'renewable']
    count = 0
    for a in activities:
        # Use mode with minimum total renewable demand
        min_mode = min(a['modes'],
                       key=lambda m: sum(m['renewable_demands'].values()))
        for rid in renewable_ids:
            if min_mode['renewable_demands'].get(rid, 0) > 0:
                count += 1
    return count / (n * K)


def _compute_RS(activities: list, resources: list,
                horizon: int) -> float:
    """
    Resource Strength: how tight resource capacities are.
    RS = min_k [ C_k / max_t sum_j d_jk * x_jt ]
    Approximated as: RS = min_k [ C_k / avg_peak_demand_k ]
    Range [0,1]. Low RS → tight resources → harder.
    Reference: Kolisch & Sprecher 1997.
    """
    rs_values = []
    for r in resources:
        cap = r['capacity']
        total_demand = sum(
            a['demands'].get(r['id'], 0) for a in activities)
        if total_demand == 0:
            rs_values.append(1.0)
            continue
        # Average demand if all activities ran in parallel
        avg_parallel = total_demand / max(1, len(activities))
        rs_values.append(min(1.0, cap / avg_parallel))
    return min(rs_values) if rs_values else 1.0


def _compute_RS_multimode(activities: list, resources: list) -> float:
    """RS for multi-mode: use minimum-demand mode per activity."""
    rs_values = []
    renewable = [r for r in resources if r['type'] == 'renewable']
    for r in renewable:
        cap = r['capacity']
        total = sum(
            min(m['renewable_demands'].get(r['id'], 0)
                for m in a['modes'])
            for a in activities)
        if total == 0:
            rs_values.append(1.0)
            continue
        avg = total / max(1, len(activities))
        rs_values.append(min(1.0, cap / avg))
    return min(rs_values) if rs_values else 1.0


def _compute_critical_path(activities: list,
                           precedences: list) -> int:
    """
    Longest path (critical path length) via topological sort + DP.
    Uses durations only (no resource constraints).
    """
    dur = {a['id']: a['duration'] for a in activities}
    succ = {a['id']: [] for a in activities}
    pred = {a['id']: [] for a in activities}
    for p in precedences:
        succ[p['i']].append(p['j'])
        pred[p['j']].append(p['i'])

    # Kahn's algorithm topological sort
    in_degree = {a['id']: len(pred[a['id']]) for a in activities}
    queue = [aid for aid, d in in_degree.items() if d == 0]
    est = {a['id']: 0 for a in activities}  # earliest start

    while queue:
        node = queue.pop(0)
        for s in succ[node]:
            est[s] = max(est[s], est[node] + dur[node])
            in_degree[s] -= 1
            if in_degree[s] == 0:
                queue.append(s)

    return max(est[a['id']] + dur[a['id']] for a in activities)


def _compute_critical_path_multimode(activities: list,
                                     precedences: list) -> int:
    """CP for multi-mode: use minimum duration mode per activity."""
    single_mode = [
        {"id": a['id'],
         "duration": min(m['duration'] for m in a['modes']),
         "demands": {}}
        for a in activities
    ]
    return _compute_critical_path(single_mode, precedences)


def _compute_network_density(activities: list,
                             precedences: list) -> float:
    """
    Network density: actual edges / maximum possible edges.
    density = |E| / (n * (n-1) / 2)
    """
    n = len(activities)
    if n < 2:
        return 0.0
    return len(precedences) / (n * (n - 1) / 2)


def _compute_structural_hash(activities: list,
                             precedences: list) -> str:
    """
    SHA-256 of sorted (i,j) precedence pairs.
    Used to detect structural duplicates across the dataset.
    Invariant to activity duration/demand changes.
    """
    edges = sorted((p['i'], p['j']) for p in precedences)
    payload = json.dumps(edges, sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def _compute_structural_hash_multimode(activities: list,
                                       precedences: list) -> str:
    return _compute_structural_hash(activities, precedences)


# ══════════════════════════════════════════════════════════════════
# Graph summary header (sent to GPT-4o — never full JSON)
# ══════════════════════════════════════════════════════════════════

def build_graph_summary(graph: dict) -> str:
    """
    Produces compact ~100-token summary for GPT-4o context.
    NEVER includes individual task durations or resource demands.
    """
    meta = graph['meta']
    ctype = meta['type']
    resources = graph['resources']
    activities = graph['activities']
    precedences = graph['precedences']

    r_summary = ", ".join(
        f"{r['id']}(cap={r['capacity']})"
        for r in resources
        if r.get('type', 'renewable') == 'renewable'
    )
    nr_summary = ", ".join(
        f"{r['id']}(budget={r['capacity']})"
        for r in resources
        if r.get('type') == 'nonrenewable'
    )
    n_activities = len(activities)

    # ── Critical path: dispatch on type ──────────────────────────
    if ctype == 'TYPE-B':
        # TYPE-B activities have 'modes', no top-level 'duration'
        cp = _compute_critical_path_multimode(activities, precedences)
    else:
        # TYPE-A and TYPE-C have top-level 'duration'
        cp = _compute_critical_path(activities, precedences)

    density = _compute_network_density(activities, precedences)

    lines = [
        f"instance_id: {meta['instance_id']}",
        f"constraint_type: {ctype}",
        f"activities: {n_activities}",
        f"renewable resources: {r_summary}",
    ]
    if nr_summary:
        lines.append(f"nonrenewable resources: {nr_summary}")
    lines += [
        f"critical_path_length: {cp}",
        f"network_density: {density:.2f}",
        f"RF: {meta['RF']:.2f}   RS: {meta['RS']:.2f}",
        f"difficulty: {meta['difficulty'] or 'unset'}"
    ]
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════
# Stratifier: assigns difficulty based on RF/RS quartiles
# ══════════════════════════════════════════════════════════════════

def assign_difficulty(graph: dict,
                      rf_thresholds: tuple = (0.4, 0.65),
                      rs_thresholds: tuple = (0.35, 0.6)) -> dict:
    """
    Assigns difficulty based on RF and RS per Kolisch & Sprecher 1997.

    RF (Resource Factor): fraction of resource types used per task.
      High RF → more resource type contention → HARDER
      Low  RF → fewer resource types demanded → EASIER

    RS (Resource Strength): capacity tightness.
      Low  RS → tight capacity → HARDER
      High RS → loose capacity → EASIER

    Easy: LOW RF  (≤ rf_low)  AND HIGH RS (≥ rs_high)
    Hard: HIGH RF (≥ rf_high) AND LOW RS  (≤ rs_low)
    Medium: everything else (mixed conditions)

    Thresholds are quartile-based from PSPLIB J30/J60 distribution.
    Adjust after pilot if empirical distribution is skewed.
    """
    import copy
    graph = copy.deepcopy(graph)
    rf = graph['meta']['RF']
    rs = graph['meta']['RS']
    rf_low, rf_high = rf_thresholds  # e.g. 0.4, 0.65
    rs_low, rs_high = rs_thresholds  # e.g. 0.35, 0.6

    if rf <= rf_low and rs >= rs_high:
        # Low contention, loose capacity → easy
        difficulty = 'easy'
    elif rf >= rf_high and rs <= rs_low:
        # High contention, tight capacity → hard
        difficulty = 'hard'
    else:
        # Mixed: medium difficulty
        difficulty = 'medium'

    graph['meta']['difficulty'] = difficulty
    return graph


# ══════════════════════════════════════════════════════════════════
# Batch parser: processes all instances in a directory
# ══════════════════════════════════════════════════════════════════

def parse_all_psplib(directory: str | Path,
                     split_ratios: tuple = (0.7, 0.1, 0.2)
                     ) -> list[dict]:
    """
    Parses all .sm files in directory.
    Applies stratified split: 70% train, 10% val, 20% test.
    Stratification is on (difficulty × RF_quartile × RS_quartile)
    so each split has similar difficulty distribution.
    """
    from sklearn.model_selection import train_test_split

    files = sorted(Path(directory).glob("*.sm"))
    graphs = []
    for f in files:
        g = parse_psplib(f, instance_id=f.stem)
        g = assign_difficulty(g)
        graphs.append(g)

    # Stratify on difficulty label
    labels = [g['meta']['difficulty'] for g in graphs]
    train_g, temp_g, _, temp_l = train_test_split(
        graphs, labels,
        test_size=1 - split_ratios[0],
        stratify=labels, random_state=42)
    val_ratio_of_temp = split_ratios[1] / (split_ratios[1] + split_ratios[2])
    val_g, test_g = train_test_split(
        temp_g, test_size=1 - val_ratio_of_temp,
        stratify=temp_l, random_state=42)

    for g in train_g: g['meta']['split'] = 'train'
    for g in val_g:   g['meta']['split'] = 'val'
    for g in test_g:  g['meta']['split'] = 'test'

    return train_g + val_g + test_g


# ══════════════════════════════════════════════════════════════════
# Private helpers (Robust Version)
# ══════════════════════════════════════════════════════════════════

def _extract_int(lines: list[str], keyword: str) -> int:
    target = keyword.replace(" ", "").lower()
    for line in lines:
        # 去除所有空格进行容错匹配
        if target in line.replace(" ", "").lower():
            nums = re.findall(r'\d+', line)
            if nums:
                return int(nums[0])
    raise ValueError(f"Keyword '{keyword}' not found in file")


def _find_section(lines: list[str], section_name: str) -> int:
    target = section_name.replace(" ", "").lower()
    for i, line in enumerate(lines):
        # 兼容 "RESOURCE AVAILABILITIES" 等带空格的变体
        if target in line.replace(" ", "").lower():
            return i
    raise ValueError(f"Section '{section_name}' not found")


def _read_data_lines(lines: list[str], start: int) -> list[str]:
    """鲁棒地读取数据块：自动跳过表头文字和 ***/--- 分隔符"""
    result = []
    in_data = False
    for line in lines[start:]:
        line_clean = line.strip()
        if not line_clean:
            continue

        # 处理边界分隔符 (* 或 -)
        if line_clean.startswith('*') or line_clean.startswith('-'):
            if in_data:
                break  # 已经读完数据，遇到闭合分隔符则结束
            continue  # 还没读到数据，遇到顶部跳过即可

        # 跳过纯文本表头 (例如 'jobnr.', 'R 1', 'mode' 等)
        if re.match(r'^[a-zA-Z]', line_clean):
            continue

        # 剩下的就是纯净的数字数据了
        in_data = True
        result.append(line_clean)

    return result
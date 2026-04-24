"""
validator.py — Module 3
Pre-application validation of mutation specs.
Post-application feasibility checking of mutated graphs.
Distance graph builder with correct BMR 1988 edge weights.
Bellman-Ford negative-weight cycle detection.

Depends on: instance_parser.py (schema), graph_mutator.py (apply_mutations for simulation)
"""
import copy
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Optional


@dataclass
class ValidationResult:
    valid: bool
    reason: Optional[str] = None


# ══════════════════════════════════════════════════════════════════
# ALLOWED OPS PER TYPE (enforced policy layer)
# ══════════════════════════════════════════════════════════════════

ALLOWED_OPS = {
    "TYPE-A": {
        "reduce_capacity", "freeze_resource",
        "enforce_time_lag",  # lag_max must be None for TYPE-A
        "add_precedence", "alter_objective"
    },
    "TYPE-B": {
        "alter_mode_profile", "restrict_modes",
        "add_mode", "reduce_nonrenewable_budget",
        "reduce_capacity",  # renewable resources can still be reduced
        "alter_objective"
    },
    "TYPE-C": {
        "reduce_capacity", "freeze_resource",
        "enforce_time_lag",  # lag_max MAY be set for TYPE-C
        "add_precedence", "alter_objective"
    }
}


# ══════════════════════════════════════════════════════════════════
# DURATION LOOKUP HELPER
# Handles both TYPE-A (top-level duration) and TYPE-B (min-mode)
# ══════════════════════════════════════════════════════════════════
def get_duration(activity: dict) -> int:
    """
    Returns activity duration for distance graph computation.
    TYPE-A/C: activity has top-level 'duration' field.
    TYPE-B:   activity has 'modes' list — use minimum duration
              (most optimistic, gives tightest feasibility check).
    """
    if 'duration' in activity:
        return activity['duration']
    elif 'modes' in activity:
        return min(m['duration'] for m in activity['modes'])
    else:
        raise KeyError(
            f"Activity {activity['id']} has neither 'duration' "
            f"nor 'modes' field")


def build_duration_map(graph: dict) -> dict[int, int]:
    """Returns {activity_id: duration} for all activities."""
    return {a['id']: get_duration(a) for a in graph['activities']}


# ══════════════════════════════════════════════════════════════════
# DISTANCE GRAPH BUILDER (Bartusch, Möhring & Radermacher 1988)
# All nodes = start times S_i. Durations absorbed into edge weights.
# ══════════════════════════════════════════════════════════════════
def build_distance_graph(graph: dict) -> dict:
    """
    Converts precedence constraints to a weighted directed graph
    where each node represents an activity's start time S_i.

    Returns: {
        'nodes': set of activity ids,
        'edges': list of (from, to, weight) tuples
    }

    Edge weight derivation (all constraints normalized to S_j >= S_i + W):

    Forward edges (minimum lag):
      FS:  S_j >= End_i + lmin = S_i + d_i + lmin    W = d_i + lmin
      SS:  S_j >= S_i + lmin                          W = lmin
      FF:  End_j >= End_i + lmin
           S_j + d_j >= S_i + d_i + lmin
           S_j >= S_i + (d_i - d_j + lmin)            W = d_i - d_j + lmin
      SF:  End_j >= S_i + lmin
           S_j + d_j >= S_i + lmin
           S_j >= S_i + (lmin - d_j)                  W = lmin - d_j
    Backward edges (maximum lag — expressed as S_i >= S_j + W):
      FS:  S_j <= End_i + lmax = S_i + d_i + lmax
           S_i >= S_j - (d_i + lmax)                  W = -(d_i + lmax)
      SS:  S_j <= S_i + lmax
           S_i >= S_j - lmax                           W = -lmax
      FF:  End_j <= End_i + lmax
           S_j + d_j <= S_i + d_i + lmax
           S_i >= S_j + (d_j - d_i - lmax)            W = d_j - d_i - lmax
      SF:  End_j <= S_i + lmax
           S_j + d_j <= S_i + lmax
           S_i >= S_j + (d_j - lmax)                  W = d_j - lmax

    Negative-weight cycle in this graph = infeasible constraint set.
    Reference: Bartusch, Möhring & Radermacher (1988)
    """
    dur = build_duration_map(graph)
    nodes = set(dur.keys())
    edges = []
    for p in graph['precedences']:
        i, j = p['i'], p['j']
        rel = p['relation']
        lmin = p['lag_min']
        lmax = p.get('lag_max')
        di = dur[i]
        dj = dur[j]

        # ── Forward edge i→j (minimum lag constraint) ─────────
        if rel == 'FS':
            edges.append((i, j, di + lmin))
        elif rel == 'SS':
            edges.append((i, j, lmin))
        elif rel == 'FF':
            edges.append((i, j, di - dj + lmin))
        elif rel == 'SF':
            edges.append((i, j, lmin - dj))

        # ── Backward edge j→i (maximum lag constraint) ────────
        if lmax is not None:
            if rel == 'FS':
                edges.append((j, i, -(di + lmax)))
            elif rel == 'SS':
                edges.append((j, i, -lmax))
            elif rel == 'FF':
                edges.append((j, i, dj - di - lmax))
            elif rel == 'SF':
                edges.append((j, i, dj - lmax))
    return {'nodes': nodes, 'edges': edges}


# ══════════════════════════════════════════════════════════════════
# BELLMAN-FORD NEGATIVE CYCLE DETECTION
# ══════════════════════════════════════════════════════════════════
def has_infeasible_cycle(dist_graph: dict) -> bool:
    """
    In BMR 1988 distance graph convention:
    edge i→j weight W means S_j >= S_i + W.

    Infeasibility = positive-weight cycle:
    S_i >= S_i + W where W > 0 → impossible.

    To detect with standard Bellman-Ford (which finds
    negative cycles): negate all edge weights, then
    a positive cycle becomes a negative cycle.

    Equivalently: reverse the relaxation direction.
    Instead of dist[v] = min(dist[u] + w), use
    dist[v] = max(dist[u] + w) — longest path detection.
    A cycle is infeasible if the longest-path computation
    does not converge (i.e., values keep increasing).
    """
    nodes = dist_graph['nodes']
    edges = list(dist_graph['edges'])

    SOURCE = '__source__'
    all_nodes = nodes | {SOURCE}
    # 🚨 必须加回这一步：将虚拟源点连接到所有真实节点！
    for n in nodes:
        edges.append((SOURCE, n, 0))
    # Longest-path Bellman-Ford (detect positive-weight cycles)
    NEGINF = float('-inf')
    dist = {v: NEGINF for v in all_nodes}
    dist[SOURCE] = 0

    n_nodes = len(all_nodes)

    # |V|-1 relaxation rounds — MAXIMIZE instead of minimize
    for _ in range(n_nodes - 1):
        updated = False
        for u, v, w in edges:
            if dist[u] != NEGINF and dist[u] + w > dist[v]:
                dist[v] = dist[u] + w
                updated = True
        if not updated:
            break

    # Round |V|: if any value can still INCREASE → positive cycle
    for u, v, w in edges:
        if dist[u] != NEGINF and dist[u] + w > dist[v]:
            return True  # positive-weight cycle = infeasible

    return False


# ══════════════════════════════════════════════════════════════════
# TOPOLOGICAL SORT (DAG check — only for pure FS/SS without lag_max)
# ══════════════════════════════════════════════════════════════════
def is_dag(graph: dict) -> bool:
    """
    Kahn's algorithm: attempts topological sort.
    Returns True if graph is a valid DAG (no cycles).
    Only correct for graphs where all edge weights are non-negative
    (pure FS/SS with lag_min >= 0 and no lag_max).
    """
    activity_ids = {a['id'] for a in graph['activities']}
    in_degree = {aid: 0 for aid in activity_ids}
    successors = defaultdict(list)

    for p in graph['precedences']:
        if p['i'] in activity_ids and p['j'] in activity_ids:
            successors[p['i']].append(p['j'])
            in_degree[p['j']] += 1

    queue = deque(aid for aid, deg in in_degree.items() if deg == 0)
    visited = 0

    while queue:
        node = queue.popleft()
        visited += 1
        for s in successors[node]:
            in_degree[s] -= 1
            if in_degree[s] == 0:
                queue.append(s)

    return visited == len(activity_ids)


# ══════════════════════════════════════════════════════════════════
# CYCLE DETECTION METHOD SELECTOR
# ════════════════════════════════════════════════════════════════
def needs_bellman_ford(graph: dict) -> bool:
    """
    Conservative rule: use Bellman-Ford if ANY of:
    1. Any precedence has lag_max != None (backward edge, always negative)
    2. Any FF relation (forward weight = d_i - d_j + lmin, can be negative)
    3. Any SF relation (forward weight = lmin - d_j, almost always negative)

    For pure FS/SS with lag_min >= 0 and no lag_max:
      FS forward = d_i + lmin >= 0 always
      SS forward = lmin >= 0 always
      → all edges non-negative → topological sort is sufficient
    """
    for p in graph['precedences']:
        if p.get('lag_max') is not None:
            return True
        if p['relation'] in ('FF', 'SF'):
            return True
    return False


def check_cycle_free(graph: dict) -> ValidationResult:
    if needs_bellman_ford(graph):
        dist_graph = build_distance_graph(graph)
        if has_infeasible_cycle(dist_graph):
            return ValidationResult(False,
                                    "Positive-weight cycle detected in distance graph "
                                    "(longest-path Bellman-Ford, per BMR 1988). "
                                    "Constraint set is infeasible.")
    else:
        if not is_dag(graph):
            return ValidationResult(False,
                                    "Precedence cycle detected (topological sort).")
    return ValidationResult(True)


# ══════════════════════════════════════════════════════════════════
# PRE-APPLICATION VALIDATOR
# Runs on mutation spec + original graph BEFORE GraphMutator.
# ══════════════════════════════════════════════════════════════════
def validate_mutation_spec(spec_list: list[dict],
                           graph: dict,
                           resolved_specs: list[dict]
                           ) -> ValidationResult:
    """
    Validates a list of resolved mutation specs against the graph.
    Steps:
    1. Reference validity (all IDs exist)
    2. Op allowlist per type
    3. Parameter bound checks
    4. TYPE-A lag_max restriction
    5. Nonrenewable feasibility for restrict_modes
    6. Cycle detection on simulated graph (after all mutations applied)

    Arguments:
      spec_list:      original GPT-4o mutation specs (for logging)
      graph:          original unmutated internal graph JSON
      resolved_specs: specs after severity resolver (concrete params)

    Returns ValidationResult(valid, reason).
    """
    ctype = graph['meta']['type']
    activity_ids = {a['id'] for a in graph['activities']}
    resource_ids = {r['id'] for r in graph['resources']}
    allowed = ALLOWED_OPS.get(ctype, set())

    for idx, params in enumerate(resolved_specs):
        op = params['op']

        # ── 1. Op allowlist ───────────────────────────────────
        if op not in allowed:
            return ValidationResult(False,
                                    f"Mutation #{idx}: op '{op}' not allowed "
                                    f"for {ctype}")

        # ── 2. Reference validity ─────────────────────────────
        for field in ('activity_id', 'i', 'j'):
            if field in params:
                if params[field] not in activity_ids:
                    return ValidationResult(False,
                                            f"Mutation #{idx}: unknown activity "
                                            f"id {params[field]}")
        if 'resource' in params:
            if params['resource'] not in resource_ids:
                return ValidationResult(False,
                                        f"Mutation #{idx}: unknown resource "
                                        f"id '{params['resource']}'")

                # ── 3. Parameter bound checks ─────────────────────────
        if op in ('reduce_capacity', 'freeze_resource'):
            cap = next(
                r['capacity'] for r in graph['resources']
                if r['id'] == params['resource'])
            if params['delta'] <= 0:
                return ValidationResult(False,
                                        f"Mutation #{idx}: delta must be > 0")
            if op == 'reduce_capacity' and params['delta'] >= cap:
                return ValidationResult(False,
                                        f"Mutation #{idx}: delta ({params['delta']}) "
                                        f">= capacity ({cap}). "
                                        f"Use freeze_resource for full shutdown.")
            if params['t_start'] >= params['t_end']:
                return ValidationResult(False,
                                        f"Mutation #{idx}: t_start >= t_end")

        elif op == 'enforce_time_lag':
            rel = params.get('relation', 'FS')
            if rel not in {'FS', 'SS', 'FF', 'SF'}:
                return ValidationResult(False,
                                        f"Mutation #{idx}: unknown relation '{rel}'")
            if params['lag_min'] < 0:
                return ValidationResult(False,
                                        f"Mutation #{idx}: lag_min must be >= 0")
            if params.get('lag_max') is not None:
                if params['lag_max'] <= params['lag_min']:
                    return ValidationResult(False,
                                            f"Mutation #{idx}: lag_max "
                                            f"({params['lag_max']}) must be > "
                                            f"lag_min ({params['lag_min']})")

            # ── 4. TYPE-A restriction: no lag_max allowed ─────
            if ctype == 'TYPE-A' and params.get('lag_max') is not None:
                return ValidationResult(False,
                                        f"Mutation #{idx}: lag_max is not allowed "
                                        f"for TYPE-A (RCPSP). "
                                        f"Only TYPE-C (RCPSP/max) supports lag_max.")

        elif op == 'alter_mode_profile':
            aid = params['activity_id']
            mid = params['mode_id']
            activity = next(
                (a for a in graph['activities'] if a['id'] == aid),
                None)
            if activity is None:
                return ValidationResult(False,
                                        f"Mutation #{idx}: activity {aid} not found")
            mode = next(
                (m for m in activity['modes'] if m['mode_id'] == mid),
                None)
            if mode is None:
                return ValidationResult(False,
                                        f"Mutation #{idx}: mode {mid} not found "
                                        f"in activity {aid}")
            new_dur = mode['duration'] + params.get('duration_delta', 0)
            if new_dur <= 0:
                return ValidationResult(False,
                                        f"Mutation #{idx}: resulting duration "
                                        f"{new_dur} <= 0")

        elif op == 'restrict_modes':
            aid = params['activity_id']
            activity = next(
                (a for a in graph['activities'] if a['id'] == aid),
                None)
            if activity is None:
                return ValidationResult(False,
                                        f"Mutation #{idx}: activity {aid} not found")
            remaining = [
                m for m in activity['modes']
                if m['mode_id'] not in params['remove_mode_ids']]
            if len(remaining) < 1:
                return ValidationResult(False,
                                        f"Mutation #{idx}: restricting modes "
                                        f"{params['remove_mode_ids']} on activity "
                                        f"{aid} leaves 0 modes")

            # ── 5. Nonrenewable feasibility after restriction ──
            # Simulate this single restriction on a copy
            # and check NR budget feasibility
            sim_graph = copy.deepcopy(graph)
            for a in sim_graph['activities']:
                if a['id'] == aid:
                    a['modes'] = [
                        m for m in a['modes']
                        if m['mode_id'] not in params['remove_mode_ids']
                    ]
                    break
            for r in sim_graph['resources']:
                if r.get('type') != 'nonrenewable':
                    continue
                min_total_demand = sum(
                    min(m['nonrenewable_demands'].get(r['id'], 0)
                        for m in a['modes'])
                    for a in sim_graph['activities']
                )
                if min_total_demand > r['capacity']:
                    return ValidationResult(False,
                                            f"Mutation #{idx}: restricting modes on "
                                            f"activity {aid} makes nonrenewable "
                                            f"resource {r['id']} infeasible "
                                            f"(min demand {min_total_demand} > "
                                            f"budget {r['capacity']})")

        elif op == 'reduce_nonrenewable_budget':
            rid = params['resource']
            r = next(
                (r for r in graph['resources'] if r['id'] == rid),
                None)
            if r is None:
                return ValidationResult(False,
                                        f"Mutation #{idx}: resource {rid} not found")
            if r.get('type') != 'nonrenewable':
                return ValidationResult(False,
                                        f"Mutation #{idx}: {rid} is not nonrenewable")
            if params['delta'] <= 0:
                return ValidationResult(False,
                                        f"Mutation #{idx}: delta must be > 0")
            if params['delta'] >= r['capacity']:
                return ValidationResult(False,
                                        f"Mutation #{idx}: delta ({params['delta']}) "
                                        f">= budget ({r['capacity']})")
            remaining = r['capacity'] - params['delta']
            min_demand = sum(
                min(m['nonrenewable_demands'].get(rid, 0)
                    for m in a['modes'])
                for a in graph['activities']
            )
            if min_demand > remaining:
                return ValidationResult(False,
                                        f"Mutation #{idx}: reducing {rid} budget to "
                                        f"{remaining} makes problem infeasible "
                                        f"(min mode demand total = {min_demand})")

    # ── 6. Cycle detection on simulated graph ─────────────────
    # Apply all mutations to a copy and check for cycles
    from graph_mutator import apply_mutations
    try:
        simulated = apply_mutations(graph, resolved_specs)
    except AssertionError as e:
        return ValidationResult(False,
                                f"Mutation application failed: {e}")

    cycle_result = check_cycle_free(simulated)
    if not cycle_result.valid:
        return cycle_result

    return ValidationResult(True)


# ══════════════════════════════════════════════════════════════════
# POST-APPLICATION FEASIBILITY CHECK
# Runs on mutated graph AFTER GraphMutator.
# Catches valid individual mutations that combine badly.
# This is a lightweight pre-check before Gate 4 (CP-SAT 10s solve).
# ══════════════════════════════════════════════════════════════════
def post_application_check(mutated_graph: dict) -> ValidationResult:
    """
    Lightweight check on the fully mutated graph.
    Runs AFTER GraphMutator, BEFORE CP-SAT oracle.
    Catches:
    - Precedence cycles / negative-weight cycles
    - Resources with all capacity consumed by downtime at every timeslot
      (resource effectively unusable but tasks demand it)

    This is NOT the full feasibility check — that is the CP-SAT
    oracle (Gate 4, 10s time limit). This catches provably infeasible
    cases cheaply to avoid wasting oracle time.
    """
    # 1. Cycle check (Bellman-Ford or topological sort)
    cycle_result = check_cycle_free(mutated_graph)
    if not cycle_result.valid:
        return cycle_result

    # 2. Resource usability check: if downtime covers entire horizon,
    #    resource is effectively dead but tasks may still demand it
    dur_map = build_duration_map(mutated_graph)
    total_duration = sum(dur_map.values())
    # Rough horizon estimate (safe upper bound)
    lag_sum = sum(
        p['lag_min'] for p in mutated_graph['precedences']
        if p.get('lag_min', 0) > 0)
    horizon_est = total_duration + lag_sum

    for r in mutated_graph['resources']:
        if r.get('type') == 'nonrenewable':
            continue
        downtime = r.get('downtime', [])
        if not downtime:
            continue
        # Check if combined downtime covers entire horizon
        # with delta >= capacity (full shutdown)
        full_shutdown_coverage = sum(
            dt['end'] - dt['start']
            for dt in downtime
            if dt['delta'] >= r['capacity']
        )
        if full_shutdown_coverage >= horizon_est:
            # Check if any task actually demands this resource
            has_demand = False
            for a in mutated_graph['activities']:
                if 'demands' in a:
                    if a['demands'].get(r['id'], 0) > 0:
                        has_demand = True
                        break
                elif 'modes' in a:
                    if any(m['renewable_demands'].get(r['id'], 0) > 0
                           for m in a['modes']):
                        has_demand = True
                        break
            if has_demand:
                return ValidationResult(False,
                                        f"Resource {r['id']} is fully shut down "
                                        f"for the entire horizon but tasks demand it")

    return ValidationResult(True)
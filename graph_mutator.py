"""
ggraph_mutator.py — Module 2
Applies validated, resolved mutation operations to internal graph JSON.
All functions are pure (deep-copy input, return new graph).
Depends on: instance_parser.py (schema only, not imported directly).
"""
import copy, random, math
from typing import Any


# ══════════════════════════════════════════════════════════════════
# SEVERITY RESOLVER
# Called BEFORE validator and GraphMutator.
# Converts GPT-4o severity labels → concrete numerical parameters.
# ══════════════════════════════════════════════════════════════════


def resolve_severity(mut_spec: dict, graph: dict) -> dict:
    """
    Takes a single mutation spec entry from GPT-4o output
    (with severity labels) and resolves it to concrete params.
    Returns a new dict with all numerical values filled in.
    h = critical_path_length from graph meta (set by parser).
    """
    op = mut_spec['op']
    cp = graph['meta'].get('critical_path_length', 100)
    resolved = copy.deepcopy(mut_spec)

    if op in ('reduce_capacity', 'freeze_resource'):
        rid = mut_spec['resource']
        cap = next(r['capacity'] for r in graph['resources']
                   if r['id'] == rid)
        sev = mut_spec.get('severity', 'moderate')
        delta_map = {
            'minor': 1,
            'moderate': 2,
            'major': max(1, cap // 2),
            'critical': max(1, int(cap * 0.9))
        }
        resolved['delta'] = delta_map[sev]
        if op == 'freeze_resource':
            resolved['delta'] = cap  # full shutdown

        # Window length from severity (fraction of critical path)
        win_map = {
            'minor': max(1, cp // 8),
            'moderate': max(1, cp // 5),
            'major': max(1, cp // 3),
            'critical': max(1, cp // 2)
        }
        win = win_map[sev]

        # t_start from positional label
        pos = mut_spec.get('t_start', 'mid')
        pos_map = {
            'early': lambda: random.randint(0, max(0, cp // 5)),
            'mid': lambda: random.randint(2 * cp // 5, 3 * cp // 5),
            'late': lambda: random.randint(7 * cp // 10, 9 * cp // 10)
        }
        t_start = pos_map.get(pos, pos_map['mid'])()
        resolved['t_start'] = t_start
        resolved['t_end'] = t_start + win

    elif op == 'enforce_time_lag':
        sev_min = mut_spec.get('lag_severity', 'moderate')
        lag_min_map = {
            'minor': 1, 'moderate': 3,
            'major': 8, 'critical': 24
        }
        lag_min = lag_min_map[sev_min]
        resolved['lag_min'] = lag_min

        if mut_spec.get('has_max_lag', False):
            # lag_max = lag_min + extra_window
            # NARROWER window = HARDER = more CRITICAL (corrected)
            sev_max = mut_spec.get('max_lag_severity', 'moderate')
            extra_map = {
                'minor': 48,  # wide window  → easy constraint
                'moderate': 24,
                'major': 12,
                'critical': 4  # tight window → hard constraint
            }
            resolved['lag_max'] = lag_min + extra_map[sev_max]
        else:
            resolved['lag_max'] = None

        # relation defaults to FS if not specified
        if 'relation' not in resolved:
            resolved['relation'] = 'FS'

    elif op == 'alter_mode_profile':
        sev = mut_spec.get('severity', 'moderate')
        dur_delta_map = {
            'minor': 1, 'moderate': 2,
            'major': 4, 'critical': 6
        }
        resolved['duration_delta'] = dur_delta_map[sev]

    elif op == 'reduce_nonrenewable_budget':
        rid = mut_spec['resource']
        budget = next(r['capacity'] for r in graph['resources']
                      if r['id'] == rid)
        sev = mut_spec.get('severity', 'moderate')
        delta_map = {
            'minor': max(1, budget // 10),
            'moderate': max(1, budget // 5),
            'major': max(1, budget // 3),
            'critical': max(1, budget // 2)
        }
        resolved['delta'] = delta_map[sev]

    elif op == 'alter_objective':
        # No severity to resolve — params passed directly
        pass

    return resolved


# ══════════════════════════════════════════════════════════════════
# HELPER: apply all mutations in a spec list
# ══════════════════════════════════════════════════════════════════

def apply_mutations(graph: dict,
                    resolved_specs: list[dict]) -> dict:
    """
    Applies a list of resolved mutation specs in order.
    Returns the fully mutated graph.
    Validator must have already approved the full spec list.
    """
    g = copy.deepcopy(graph)
    for spec in resolved_specs:
        op = spec['op']
        if op == 'reduce_capacity':
            g = reduce_capacity(g, spec['resource'],
                                spec['delta'],
                                spec['t_start'], spec['t_end'])
        elif op == 'freeze_resource':
            g = freeze_resource(g, spec['resource'],
                                spec['t_start'], spec['t_end'])
        elif op == 'enforce_time_lag':
            g = enforce_time_lag(g, spec['i'], spec['j'],
                                 spec['relation'],
                                 spec['lag_min'],
                                 spec.get('lag_max'))
        elif op == 'add_precedence':
            g = add_precedence(g, spec['i'], spec['j'],
                               spec.get('relation', 'FS'))
        elif op == 'alter_objective':
            g = alter_objective(g, spec['obj_type'],
                                spec.get('task_weights'),
                                spec.get('task_deadlines'))
        elif op == 'alter_mode_profile':
            g = alter_mode_profile(g, spec['activity_id'],
                                   spec['mode_id'],
                                   spec['duration_delta'],
                                   spec.get('renewable_demand_deltas', {}),
                                   spec.get('nonrenewable_demand_deltas', {}))
        elif op == 'restrict_modes':
            g = restrict_modes(g, spec['activity_id'],
                               spec['remove_mode_ids'])
        elif op == 'add_mode':
            g = add_mode(g, spec['activity_id'],
                         spec['duration'],
                         spec['renewable_demands'],
                         spec['nonrenewable_demands'])
        elif op == 'reduce_nonrenewable_budget':
            g = reduce_nonrenewable_budget(g, spec['resource'],
                                           spec['delta'])
        else:
            raise ValueError(f"Unknown mutation op: {op}")

        # Record mutation in audit log
        g['mutations_applied'].append(copy.deepcopy(spec))

    return g


# ══════════════════════════════════════════════════════════════════
# TYPE-A MUTATION OPS
# ══════════════════════════════════════════════════════════════════

def reduce_capacity(graph: dict, resource_id: str,
                    delta: int, t_start: int,
                    t_end: int) -> dict:
    """
    Reduces resource capacity by delta during [t_start, t_end].
    In target m: adds entry to resource.downtime array.
    Pre-conditions checked by validator:
      delta > 0, delta < capacity, t_start < t_end
    Post-condition: downtime entry added with correct delta.
    """
    g = copy.deepcopy(graph)
    for r in g['resources']:
        if r['id'] == resource_id:
            r.setdefault('downtime', [])
            # Merge overlapping windows if same delta
            r['downtime'].append({
                'start': t_start,
                'end': t_end,
                'delta': delta
            })
            # Sort windows chronologically for clean serialization
            r['downtime'].sort(key=lambda w: w['start'])
            break
    return g


def freeze_resource(graph: dict, resource_id: str,
                    t_start: int, t_end: int) -> dict:
    """
    Full shutdown: delta = resource capacity.
    Delegates to reduce_capacity with delta = full capacity.
    """
    cap = next(r['capacity'] for r in graph['resources']
               if r['id'] == resource_id)
    return reduce_capacity(graph, resource_id, cap,
                           t_start, t_end)


def enforce_time_lag(graph: dict, i: int, j: int,
                     relation: str, lag_min: int,
                     lag_max: int | None) -> dict:
    """
    Adds or updates a precedence constraint i→j.

    ADDITIVE RULE (per RCPSP/max standard):
    - If (i,j) already has a constraint with the SAME relation type:
      update that constraint's lag_min/lag_max (tighten parameters).
    - If (i,j) has NO constraint with this relation type:
      ADD a new constraint. The existing constraints of other
      relation types remain untouched.

    This preserves the physical meaning of each constraint.
    A pair (i,j) can have simultaneous FS and SS constraints.

    Example:
      Base: (1,3,FS,lag_min=0,lag_max=None)
      Mutation: enforce_time_lag(g,1,3,'SS',0,5)
      Result: BOTH constraints exist:
        (1,3,FS,lag_min=0,lag_max=None) — original, untouched
        (1,3,SS,lag_min=0,lag_max=5)    — added by mutation
    """
    g = copy.deepcopy(graph)
    assert relation in {'FS', 'SS', 'FF', 'SF'}, \
        f"Unknown relation: {relation}"
    assert lag_min >= 0, "lag_min must be >= 0"
    if lag_max is not None:
        assert lag_max > lag_min, \
            f"lag_max ({lag_max}) must be > lag_min ({lag_min})"

    # Search for existing constraint with SAME (i, j, relation) triple
    existing = None
    for p in g['precedences']:
        if p['i'] == i and p['j'] == j and p['relation'] == relation:
            existing = p
            break

    if existing is not None:
        # Same relation type already exists → tighten parameters
        # Tighten = take the more restrictive value
        existing['lag_min'] = max(existing['lag_min'], lag_min)
        if lag_max is not None:
            if existing['lag_max'] is None:
                existing['lag_max'] = lag_max
            else:
                existing['lag_max'] = min(existing['lag_max'], lag_max)
        # Post-condition: lag_max > lag_min still holds
        if existing['lag_max'] is not None:
            assert existing['lag_max'] > existing['lag_min'], \
                (f"Tightening created invalid window: "
                 f"lag_min={existing['lag_min']}, "
                 f"lag_max={existing['lag_max']}")
    else:
        # Different relation type or new pair → ADD new constraint
        # Existing constraints for this pair remain untouched
        g['precedences'].append({
            'i': i, 'j': j,
            'relation': relation,
            'lag_min': lag_min,
            'lag_max': lag_max
        })

    return g


def add_precedence(graph: dict, i: int, j: int,
                   relation: str = 'FS') -> dict:
    """
    Adds a new precedence i→j with lag_min=0, lag_max=None.
    Idempotent check: if ANY precedence entry for (i,j)
    already exists (regardless of relation type), no-op.
    This prevents accidentally re-adding a base FS constraint
    that GPT-4o redundantly specified in its mutation spec.
    To add a DIFFERENT relation type to an existing pair,
    use enforce_time_lag — which correctly handles additivity.
    """
    g = copy.deepcopy(graph)
    # Check if ANY constraint exists for this pair
    pair_exists = any(
        p['i'] == i and p['j'] == j
        for p in g['precedences']
    )
    if not pair_exists:
        g['precedences'].append({
            'i': i, 'j': j,
            'relation': relation,
            'lag_min': 0,
            'lag_max': None
        })
    return g


def alter_objective(graph: dict, obj_type: str,
                    task_weights: dict | None,
                    task_deadlines: dict | None) -> dict:
    """
    Changes the optimization objective.
    obj_type: 'minimize_makespan'
            | 'minimize_weighted_tardiness'
            | 'minimize_resource_cost'
            | 'maximize_robustness_margin'
    task_weights and task_deadlines required for tardiness only.
    """
    g = copy.deepcopy(graph)
    valid_types = {
        'minimize_makespan',
        'minimize_weighted_tardiness',
        'minimize_resource_cost',
        'maximize_robustness_margin'
    }
    assert obj_type in valid_types, \
        f"Unknown objective type: {obj_type}"
    if obj_type == 'minimize_weighted_tardiness':
        assert task_weights is not None,(
        "task_weights required for weighted_tardiness"
        )
        assert task_deadlines is not None,(
        "task_deadlines required for weighted_tardiness"
        )

        g['objective'] = {
            'type': obj_type,
            'task_weights': task_weights,
            'task_deadlines': task_deadlines
        }
        return g

    # ══════════════════════════════════════════════════════════════════
    # TYPE-B MUTATION OPS
    # ══════════════════════════════════════════════════════════════════

def alter_mode_profile(graph: dict, activity_id: int,
                        mode_id: int,
                        duration_delta: int,
                        renewable_demand_deltas: dict,
                        nonrenewable_demand_deltas: dict) -> dict:
    """
    Changes duration and/or resource demands of one mode.
    duration_delta: positive = longer, negative = shorter.
    demand_deltas: {resource_id: delta} — positive or negative.
    Post-conditions:
    resulting duration > 0
    all resulting demands >= 0
    """
    g = copy.deepcopy(graph)
    for a in g['activities']:
        if a['id'] != activity_id:
            continue
        for m in a['modes']:
            if m['mode_id'] != mode_id:
                continue
            new_dur = m['duration'] + duration_delta
            assert new_dur > 0, \
                f"alter_mode_profile: resulting duration {new_dur} <= 0"
            m['duration'] = new_dur

            for rid, delta in renewable_demand_deltas.items():
                m['renewable_demands'][rid] = max(
                    0, m['renewable_demands'].get(rid, 0) + delta)
            for rid, delta in nonrenewable_demand_deltas.items():
                m['nonrenewable_demands'][rid] = max(
                    0, m['nonrenewable_demands'].get(rid, 0) + delta)
            break
        break
    return g

def restrict_modes(graph: dict, activity_id: int,
                    remove_mode_ids: list[int]) -> dict:
    """
    Removes specified modes from an activity.
    Pre-conditions (validator):
    len(remaining_modes) >= 1
    nonrenewable budget still feasible after restriction
    Post-condition: activity has at least 1 mode.
    """
    g = copy.deepcopy(graph)
    for a in g['activities']:
        if a['id'] != activity_id:
            continue
        original_count = len(a['modes'])
        a['modes'] = [m for m in a['modes']
                        if m['mode_id'] not in remove_mode_ids]
        assert len(a['modes']) >= 1, \
            (f"restrict_modes: activity {activity_id} has 0 modes "
             f"after removing {remove_mode_ids}")
        break
    return g

def add_mode(graph: dict, activity_id: int,
             duration: int,
             renewable_demands: dict,
             nonrenewable_demands: dict) -> dict:
    """
    Adds a new execution mode to an activity.
    New mode_id = max(existing mode_ids) + 1.
    Pre-conditions: duration > 0, all demands >= 0.
    """
    g = copy.deepcopy(graph)
    assert duration > 0, "add_mode: duration must be > 0"
    assert all(v >= 0 for v in renewable_demands.values()), \
        "add_mode: all renewable demands must be >= 0"
    assert all(v >= 0 for v in nonrenewable_demands.values()), \
        "add_mode: all nonrenewable demands must be >= 0"

    for a in g['activities']:
        if a['id'] != activity_id:
            continue
        new_mode_id = max(m['mode_id'] for m in a['modes']) + 1
        a['modes'].append({
             'mode_id': new_mode_id,
             'duration': duration,
             'renewable_demands': renewable_demands,
             'nonrenewable_demands': nonrenewable_demands
        })
        break
    return g

def reduce_nonrenewable_budget(graph: dict,
                               resource_id: str,
                               delta: int) -> dict:
    """
    Reduces a nonrenewable resource's total budget by delta.
    Pre-conditions (validator):
    delta > 0, delta < budget,
    remaining budget >= sum of minimum mode NR demands
    """
    g = copy.deepcopy(graph)
    for r in g['resources']:
        if r['id'] == resource_id:
            assert r.get('type') == 'nonrenewable', \
                f"{resource_id} is not a nonrenewable resource"
            assert delta > 0
            assert delta < r['capacity'], \
                "reduce_nonrenewable_budget: delta >= budget"
            r['capacity'] -= delta
            break
    return g
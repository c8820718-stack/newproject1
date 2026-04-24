"""
orchestrator.py — End-to-End Data Generation Pipeline
======================================================
Connects all 5 modules + GPT-4o contextualizer into a single
executable pipeline. Processes PSPLIB/MMLIB instances → validated
(p, m, c) training triplets written to JSONL.

Pipeline flow per instance:
  1. Parse base instance → internal graph JSON
  2. Build graph summary header (~100 tokens)
  3. GPT-4o B1: summary → mutation spec (severity levels)
  4. Resolve severity → concrete numerical params
  5. Validate mutation spec (pre-application)
  6. Apply mutations → mutated graph
  7. Post-application feasibility check
  8. serialize_p_structure → RESOURCES + TASKS text
  9. GPT-4o B2: structure + resolved mutations → CONTEXT + DISRUPTION
  10. Assemble full p = CONTEXT + RESOURCES + TASKS + DISRUPTION
  11. CodeBuilder → c;  serialize_target_m → target m
  12. Filter pipeline (Gates 0–7) → accept/reject
  13. Build training triplet → write to JSONL

Depends on: all modules (parser, mutator, validator, code_builder,
            filter_pipeline, gpt4o_contextualizer)
"""

import argparse
import copy
import json
import logging
import os
import random
import re
import sys
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

# Pipeline modules
from instance_parser import (
    parse_psplib, parse_mmlib, parse_all_psplib,
    assign_difficulty, build_graph_summary,
)
from graph_mutator import resolve_severity, apply_mutations
from validator import validate_mutation_spec, post_application_check
from code_builder import build_code, serialize_target_m
from filter_pipeline import (
    FilterPipeline, build_training_triplet, PipelineResult,
)
from gpt4o_contextualizer import GPT4oContextualizer

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# p TEXT SERIALIZER (deterministic — writes RESOURCES + TASKS from real data)
# ══════════════════════════════════════════════════════════════════════════════


def serialize_p_structure(mutated_graph: dict) -> str:
    """
    Reads mutated graph and produces the RESOURCES and TASKS
    subsections of p. Factually correct — no LLM involvement.
    Handles both TYPE-A (flat duration/demands) and TYPE-B (modes).
    Iterates ALL downtime windows per resource.
    """
    lines = []
    ctype = mutated_graph['meta']['type']
    resources = mutated_graph['resources']
    activities = mutated_graph['activities']
    precedences = mutated_graph['precedences']

    # ── RESOURCES ─────────────────────────────────────────────────
    lines.append("RESOURCES:")
    for r in resources:
        r_type = r.get('type', 'renewable')
        if r_type == 'nonrenewable':
            cap_str = f"budget={r['capacity']}"
        else:
            cap_str = f"cap={r['capacity']}"

        # Serialize ALL downtime windows
        if r.get('downtime'):
            window_strs = []
            for dt in r['downtime']:
                window_strs.append(
                    f"offline t=[{dt['start']},{dt['end']}] "
                    f"delta={dt['delta']}")
            cap_str += ", " + "; ".join(window_strs)

        lines.append(f"  {r['id']}({r_type},{cap_str})")

    # ── TASKS ─────────────────────────────────────────────────────
    lines.append("")
    lines.append("TASKS:")

    if ctype in ('TYPE-A', 'TYPE-C'):
        r_ids = [r['id'] for r in resources
                 if r.get('type', 'renewable') == 'renewable']
        lines.append(f"  (id, dur, {', '.join(r_ids)}):")
        for a in activities:
            demands = ",".join(
                str(a['demands'].get(rid, 0)) for rid in r_ids)
            lines.append(f"  T{a['id']}:{a['duration']}h({demands})")

    elif ctype == 'TYPE-B':
        r_ids = [r['id'] for r in resources
                 if r.get('type', 'renewable') == 'renewable']
        nr_ids = [r['id'] for r in resources
                  if r.get('type') == 'nonrenewable']
        header = f"id-mode, dur, {', '.join(r_ids)}"
        if nr_ids:
            header += f" | {', '.join(nr_ids)}"
        lines.append(f"  ({header}):")

        for a in activities:
            for m in a['modes']:
                r_dem = ",".join(
                    str(m['renewable_demands'].get(rid, 0))
                    for rid in r_ids)
                nr_dem = ",".join(
                    str(m['nonrenewable_demands'].get(nrid, 0))
                    for nrid in nr_ids)
                dem_str = r_dem
                if nr_dem:
                    dem_str += " | " + nr_dem
                lines.append(
                    f"  T{a['id']}-M{m['mode_id']}:"
                    f"{m['duration']}h({dem_str})")

    # ── PRECEDENCES ───────────────────────────────────────────────
    lines.append("  Precedences:")
    for p in precedences:
        lag_str = str(p['lag_min'])
        if p.get('lag_max') is not None:
            lag_str += f",max{p['lag_max']}"
        lines.append(
            f"  T{p['i']}-[{p['relation']},{lag_str}]->T{p['j']}")

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# p TEXT ASSEMBLER (combines GPT-4o output with serialized structure)
# ══════════════════════════════════════════════════════════════════════════════


def assemble_p(context: str, resources_tasks: str,
               disruption: str) -> str:
    """
    Assembles the full p text from its four components.
    Always in canonical order: CONTEXT → RESOURCES → TASKS → DISRUPTION.
    """
    return (
        f"CONTEXT: {context}\n\n"
        f"{resources_tasks}\n\n"
        f"DISRUPTION: {disruption}\n"
    )


# ══════════════════════════════════════════════════════════════════════════════
# SINGLE-INSTANCE PROCESSOR
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class InstanceResult:
    """Result of processing one base instance."""
    base_id: str
    success: bool
    triplet: Optional[dict] = None
    failure_stage: Optional[str] = None
    failure_reason: Optional[str] = None
    attempts: int = 0
    gpt4o_cost: float = 0.0


def process_single_instance(
        base_graph: dict,
        contextualizer: GPT4oContextualizer,
        filter_pipeline: FilterPipeline,
        question_type: str = "optimization",
        max_retries: int = 3,
) -> InstanceResult:
    """
    Processes one base instance through the full pipeline.
    Retries up to max_retries times on recoverable failures
    (GPT-4o malformed output, validator rejection, infeasibility).

    Non-recoverable failures (parse error, CodeBuilder crash)
    are not retried — they indicate a bug in the pipeline code.
    """
    base_id = base_graph['meta']['instance_id']
    ctype = base_graph['meta']['type']
    difficulty = base_graph['meta']['difficulty']
    total_cost = 0.0

    for attempt in range(1, max_retries + 1):
        try:
            # ── Step 2: Graph summary header ──────────────────
            summary = build_graph_summary(base_graph)

            # ── Step 3: GPT-4o B1 → mutation spec ────────────
            b1_result = contextualizer.generate_mutation_spec(
                graph_summary=summary,
                ctype=ctype,
                difficulty=difficulty,
                question_type=question_type)
            total_cost += b1_result.cost_usd

            if not b1_result.success:
                logger.warning(
                    f"[{base_id}] B1 failed (attempt {attempt}): "
                    f"{b1_result.error}")
                continue

            mutation_spec = b1_result.mutation_spec
            narrative_sketch = b1_result.narrative_context or ""

            # ── Step 4: Resolve severity → concrete params ────
            resolved_specs = []
            for mut in mutation_spec:
                resolved = resolve_severity(mut, base_graph)
                resolved_specs.append(resolved)

            # ── Step 5: Validate mutation spec ────────────────
            val_result = validate_mutation_spec(
                mutation_spec, base_graph, resolved_specs)
            if not val_result.valid:
                logger.info(
                    f"[{base_id}] Validator rejected (attempt {attempt}): "
                    f"{val_result.reason}")
                continue

            # ── Step 6: Apply mutations ───────────────────────
            mutated_graph = apply_mutations(base_graph, resolved_specs)

            # Set question_type in metadata
            mutated_graph['meta']['question_type'] = question_type

            # ── Step 7: Post-application check ────────────────
            post_result = post_application_check(mutated_graph)
            if not post_result.valid:
                logger.info(
                    f"[{base_id}] Post-check failed (attempt {attempt}): "
                    f"{post_result.reason}")
                continue

            # ── Step 8: Serialize RESOURCES + TASKS ───────────
            resources_tasks_text = serialize_p_structure(mutated_graph)

            # ── Step 9: GPT-4o B2 → CONTEXT + DISRUPTION ─────
            b2_result = contextualizer.generate_narrative(
                resources_tasks_text=resources_tasks_text,
                resolved_mutations=resolved_specs,
                narrative_sketch=narrative_sketch,
                ctype=ctype)
            total_cost += b2_result.cost_usd

            if not b2_result.success:
                logger.warning(
                    f"[{base_id}] B2 failed (attempt {attempt}): "
                    f"{b2_result.error}")
                continue

            # ── Step 10: Assemble full p ──────────────────────
            p_text = assemble_p(
                context=b2_result.narrative_context,
                resources_tasks=resources_tasks_text,
                disruption=b2_result.narrative_disruption)

            # ── Step 11: Build c and target m ─────────────────
            code = build_code(mutated_graph)
            target_m = serialize_target_m(mutated_graph)

            # ── Step 12: Filter pipeline (Gates 0–7) ─────────
            pipeline_result = filter_pipeline.run(
                p_text=p_text,
                mutated_graph=mutated_graph,
                code=code,
                target_m=target_m,
                base_graph=base_graph)

            if not pipeline_result.accepted:
                failed_gate = pipeline_result.failed_gate
                gate_result = pipeline_result.gate_results.get(
                    failed_gate)
                reason = gate_result.reason if gate_result else "unknown"
                logger.info(
                    f"[{base_id}] Rejected at Gate {failed_gate} "
                    f"(attempt {attempt}): {reason}")

                # Gates 0-3 are recoverable (retry with new GPT-4o call)
                # Gates 4-7 are non-recoverable for this mutation set
                if failed_gate <= 3:
                    continue
                else:
                    return InstanceResult(
                        base_id=base_id, success=False,
                        failure_stage=f"gate_{failed_gate}",
                        failure_reason=reason,
                        attempts=attempt, gpt4o_cost=total_cost)

            # ── Step 13: Build training triplet ───────────────
            triplet = build_training_triplet(
                p_text=p_text,
                target_m=target_m,
                code=code,
                pipeline_result=pipeline_result,
                graph=mutated_graph)

            return InstanceResult(
                base_id=base_id, success=True,
                triplet=triplet,
                attempts=attempt, gpt4o_cost=total_cost)

        except Exception as e:
            logger.error(
                f"[{base_id}] Unhandled exception (attempt {attempt}): "
                f"{type(e).__name__}: {e}")
            # Do not retry on code bugs — they need fixing
            return InstanceResult(
                base_id=base_id, success=False,
                failure_stage="exception",
                failure_reason=f"{type(e).__name__}: {str(e)[:200]}",
                attempts=attempt, gpt4o_cost=total_cost)

    return InstanceResult(
        base_id=base_id, success=False,
        failure_stage="max_retries",
        failure_reason=f"Failed after {max_retries} attempts",
        attempts=max_retries, gpt4o_cost=total_cost)


# ══════════════════════════════════════════════════════════════════════════════
# BATCH PROCESSOR
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class BatchStats:
    total: int = 0
    accepted: int = 0
    rejected: int = 0
    errors: int = 0
    total_cost_usd: float = 0.0
    failures_by_stage: dict = None

    def __post_init__(self):
        if self.failures_by_stage is None:
            self.failures_by_stage = {}

    def record(self, result: InstanceResult):
        self.total += 1
        self.total_cost_usd += result.gpt4o_cost
        if result.success:
            self.accepted += 1
        else:
            self.rejected += 1
            stage = result.failure_stage or 'unknown'
            self.failures_by_stage[stage] = \
                self.failures_by_stage.get(stage, 0) + 1
            if result.failure_stage == 'exception':
                self.errors += 1

    def report(self) -> dict:
        return {
            'total_processed': self.total,
            'accepted': self.accepted,
            'rejected': self.rejected,
            'errors': self.errors,
            'acceptance_rate': round(
                self.accepted / max(1, self.total), 4),
            'total_gpt4o_cost_usd': round(self.total_cost_usd, 4),
            'avg_cost_per_instance': round(
                self.total_cost_usd / max(1, self.total), 6),
            'failures_by_stage': self.failures_by_stage,
        }


QUESTION_TYPES = ['optimization', 'feasibility', 'rescheduling']
QTYPE_WEIGHTS = [0.50, 0.25, 0.25]


def run_batch(
        base_graphs: list[dict],
        output_path: str,
        n_expansions_per_instance: int = 3,
        n_solver_workers: int = 4,
        targets_per_type: int = 15000,
        gpt4o_model: str = "gpt-4o",
        max_retries_per_instance: int = 3,
) -> BatchStats:
    """
    Processes a batch of base instances through the full pipeline.
    Writes accepted triplets to JSONL output file.
    Returns BatchStats with diagnostics.
    """
    contextualizer = GPT4oContextualizer(model=gpt4o_model)
    pipeline = FilterPipeline(
        targets_per_type=targets_per_type,
        n_solver_workers=n_solver_workers)

    stats = BatchStats()
    output_file = open(output_path, 'w')

    try:
        for idx, base_graph in enumerate(base_graphs):
            base_id = base_graph['meta']['instance_id']
            ctype = base_graph['meta']['type']
            split = base_graph['meta']['split']

            # Only expand train instances (val/test get expansion only)
            n_expansions = n_expansions_per_instance
            if split in ('val', 'test'):
                n_expansions = 1

            for exp_idx in range(n_expansions):
                # Vary question type
                qtype = random.choices(
                    QUESTION_TYPES, weights=QTYPE_WEIGHTS, k=1)[0]

                result = process_single_instance(
                    base_graph=base_graph,
                    contextualizer=contextualizer,
                    filter_pipeline=pipeline,
                    question_type=qtype,
                    max_retries=max_retries_per_instance)

                stats.record(result)

                if result.success:
                    output_file.write(
                        json.dumps(result.triplet,
                                   ensure_ascii=False) + '\n')
                    output_file.flush()

                # Progress logging
                if stats.total % 50 == 0:
                    logger.info(
                        f"Progress: {stats.total} processed, "
                        f"{stats.accepted} accepted "
                        f"({stats.accepted/max(1,stats.total):.1%}), "
                        f"cost: ${stats.total_cost_usd:.2f}")

    finally:
        output_file.close()
        pipeline.shutdown()

    # Log final statistics
    final_report = stats.report()
    final_report['filter_report'] = pipeline.report()
    final_report['gpt4o_report'] = contextualizer.cost_tracker.report()

    logger.info(f"Batch complete: {json.dumps(final_report, indent=2)}")

    # Write report to companion file
    report_path = output_path.replace('.jsonl', '_report.json')
    with open(report_path, 'w') as f:
        json.dump(final_report, f, indent=2)

    return stats


# ══════════════════════════════════════════════════════════════════════════════
# PILOT RUN (50 instances per type — calibration before scaling)
# ══════════════════════════════════════════════════════════════════════════════


def run_pilot(psplib_dir: str, mmlib_dir: str,
              output_dir: str, n_per_type: int = 50):
    """
    Pilot run: processes n_per_type instances per constraint type.
    Purpose: calibrate RF/RS thresholds, measure acceptance rates,
    estimate GPT-4o costs, verify end-to-end correctness.
    """
    os.makedirs(output_dir, exist_ok=True)

    # ── Parse TYPE-A instances ────────────────────────────────────
    logger.info(f"Parsing TYPE-A instances from {psplib_dir}...")
    type_a_graphs = []
    sm_files = sorted(Path(psplib_dir).glob("*.sm"))
    for f in sm_files[:n_per_type]:
        g = parse_psplib(f, instance_id=f.stem, split="train")
        g = assign_difficulty(g)
        type_a_graphs.append(g)
    logger.info(f"Parsed {len(type_a_graphs)} TYPE-A instances")

    # ── Parse TYPE-B instances ────────────────────────────────────
    logger.info(f"Parsing TYPE-B instances from {mmlib_dir}...")
    type_b_graphs = []
    mm_files = sorted(Path(mmlib_dir).glob("*.mm"))
    for f in mm_files[:n_per_type]:
        g = parse_mmlib(f, instance_id=f.stem, split="train")
        g = assign_difficulty(g)
        type_b_graphs.append(g)
    logger.info(f"Parsed {len(type_b_graphs)} TYPE-B instances")

    # ── Log difficulty distributions ──────────────────────────────
    for label, graphs in [("TYPE-A", type_a_graphs),
                          ("TYPE-B", type_b_graphs)]:
        dist = {}
        for g in graphs:
            d = g['meta']['difficulty']
            dist[d] = dist.get(d, 0) + 1
        logger.info(f"{label} difficulty distribution: {dist}")

    # ── Run pipeline ──────────────────────────────────────────────
    all_graphs = type_a_graphs + type_b_graphs
    random.shuffle(all_graphs)

    output_path = os.path.join(output_dir, "pilot_triplets.jsonl")

    logger.info(f"Starting pilot run: {len(all_graphs)} instances "
                f"→ {output_path}")
    stats = run_batch(
        base_graphs=all_graphs,
        output_path=output_path,
        n_expansions_per_instance=1,  # 1 expansion per instance for pilot
        n_solver_workers=2,
        targets_per_type=1000,  # generous quota for pilot
        max_retries_per_instance=3)

    # ── Print pilot report ────────────────────────────────────────
    report = stats.report()
    print("\n" + "=" * 60)
    print("PILOT RUN REPORT")
    print("=" * 60)
    print(f"Total processed:   {report['total_processed']}")
    print(f"Accepted:          {report['accepted']}")
    print(f"Rejected:          {report['rejected']}")
    print(f"Errors:            {report['errors']}")
    print(f"Acceptance rate:   {report['acceptance_rate']:.1%}")
    print(f"GPT-4o cost:       ${report['total_gpt4o_cost_usd']:.2f}")
    print(f"Avg cost/instance: ${report['avg_cost_per_instance']:.4f}")
    print(f"\nFailures by stage:")
    for stage, count in sorted(report['failures_by_stage'].items()):
        print(f"  {stage}: {count}")
    print("=" * 60)

    # ── Calibration checks ────────────────────────────────────────
    acceptance = report['acceptance_rate']
    if acceptance < 0.35:
        print("\n⚠️  ACCEPTANCE RATE TOO LOW (<35%)")
        print("    Check: severity maps may be too aggressive")
        print("    Check: RF/RS thresholds may need adjustment")
        print("    Check: GPT-4o prompt may need more examples")
    elif acceptance > 0.65:
        print("\n⚠️  ACCEPTANCE RATE TOO HIGH (>65%)")
        print("    Check: filters may be too lenient")
        print("    Check: Gate 6 keyword mapper may need more phrases")
    else:
        print("\n✓  Acceptance rate in target range (35-65%)")

    # Estimate full-scale costs
    cost_per = report['avg_cost_per_instance']
    target_total = 95000  # candidates to generate
    est_cost = cost_per * target_total
    print(f"\n📊 Estimated full-scale GPT-4o cost: ${est_cost:.2f}")
    print(f"   (based on {report['total_processed']} pilot instances)")

    return stats


# ══════════════════════════════════════════════════════════════════════════════
# FULL-SCALE RUN
# ══════════════════════════════════════════════════════════════════════════════


def run_full_scale(psplib_dir: str, mmlib_dir: str,
                   output_dir: str,
                   n_solver_workers: int = 8,
                   n_expansions: int = 5):
    """
    Full-scale data generation: all PSPLIB + MMLIB instances.
    Run only AFTER successful pilot calibration.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Parse all instances with stratified splits
    logger.info("Parsing all TYPE-A instances...")
    type_a_graphs = parse_all_psplib(psplib_dir)
    logger.info(f"Parsed {len(type_a_graphs)} TYPE-A instances")

    logger.info("Parsing all TYPE-B instances...")
    type_b_graphs = []
    for f in sorted(Path(mmlib_dir).glob("*.mm")):
        g = parse_mmlib(f, instance_id=f.stem)
        g = assign_difficulty(g)
        type_b_graphs.append(g)
    logger.info(f"Parsed {len(type_b_graphs)} TYPE-B instances")

    all_graphs = type_a_graphs + type_b_graphs
    random.shuffle(all_graphs)

    output_path = os.path.join(output_dir, "training_triplets.jsonl")

    stats = run_batch(
        base_graphs=all_graphs,
        output_path=output_path,
        n_expansions_per_instance=n_expansions,
        n_solver_workers=n_solver_workers,
        targets_per_type=15000)

    return stats


# ══════════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="RCPSP Training Data Generation Pipeline")
    subparsers = parser.add_subparsers(dest='command')

    # Pilot command
    pilot_parser = subparsers.add_parser(
        'pilot', help='Run pilot (50 instances per type)')
    pilot_parser.add_argument(
        '--psplib-dir', required=True,
        help='Directory containing PSPLIB .sm files')
    pilot_parser.add_argument(
        '--mmlib-dir', required=True,
        help='Directory containing MMLIB .mm files')
    pilot_parser.add_argument(
        '--output-dir', default='output/pilot',
        help='Output directory for pilot results')
    pilot_parser.add_argument(
        '--n-per-type', type=int, default=50,
        help='Number of instances per type (default: 50)')

    # Full-scale command
    full_parser = subparsers.add_parser(
        'full', help='Run full-scale data generation')
    full_parser.add_argument(
        '--psplib-dir', required=True)
    full_parser.add_argument(
        '--mmlib-dir', required=True)
    full_parser.add_argument(
        '--output-dir', default='output/full')
    full_parser.add_argument(
        '--n-workers', type=int, default=8)
    full_parser.add_argument(
        '--n-expansions', type=int, default=5)

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                os.path.join(args.output_dir or 'output',
                             'pipeline.log')),
        ])

    if args.command == 'pilot':
        run_pilot(
            psplib_dir=args.psplib_dir,
            mmlib_dir=args.mmlib_dir,
            output_dir=args.output_dir,
            n_per_type=args.n_per_type)
    elif args.command == 'full':
        run_full_scale(
            psplib_dir=args.psplib_dir,
            mmlib_dir=args.mmlib_dir,
            output_dir=args.output_dir,
            n_solver_workers=args.n_workers,
            n_expansions=args.n_expansions)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
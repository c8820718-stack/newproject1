"""
filter_pipeline.py — Module 5 (Final Consolidated Version)
===========================================================
Orchestrates 8 filtering gates (Gate 0–7) in cost-first order.
Each gate is a standalone function returning GateResult.
Pipeline short-circuits on first failure.

Incorporates ALL corrections from design review:
  - Gate 0: semantic pre-filter on p text (heuristic, no LLM)
  - Gate 1: structural validity (handles TYPE-A and TYPE-B schemas)
  - Gate 2: syntax check via py_compile (CodeBuilder safety net)
  - Gate 3: dedup via MinHash LSH (O(1) amortized, deterministic hash)
  - Gate 4: CP-SAT feasibility via SolverPool (exec, no subprocess)
  - Gate 5: distribution quota (30/50/20 easy/medium/hard)
  - Gate 6: p↔m consistency (keyword mapper with word proximity,
            sentence-scoped time lag, dynamic resource ID patterns,
            relation-change detection)
  - Gate 7: solution quality with continuous gap-weighted loss

SolverPool: persistent worker pool using exec() in worker processes.
  - Cross-platform timeout via threading (no signal.SIGALRM)
  - Future-based result routing (no put-back loop)
  - BestObjectiveBound guarded against extreme values

Depends on: Modules 1–4 (instance_parser, graph_mutator, validator, code_builder)
"""

import hashlib
import io
import json
import math
import multiprocessing as mp
import os
import py_compile
import re
import tempfile
import threading
import time
import traceback
from collections import Counter, defaultdict
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from typing import Optional


# ══════════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class GateResult:
    passed: bool
    reason: Optional[str] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class PipelineResult:
    accepted: bool
    failed_gate: Optional[int] = None
    gate_results: dict = field(default_factory=dict)
    approx_flag: bool = False


# ══════════════════════════════════════════════════════════════════════════════
# GATE 0 — SEMANTIC PRE-FILTER (~0.1ms)
# Catches catastrophically bad GPT-4o outputs before any computation.
# Pure heuristic — no LLM call.
# ══════════════════════════════════════════════════════════════════════════════


def gate_0_semantic_prefilter(p_text: str, ctype: str) -> GateResult:
    """
    Fast heuristic checks on p text quality.
    Catches: empty, too short/long, missing sections, no task/resource refs.
    """
    # 1. Non-empty
    if not p_text or not p_text.strip():
        return GateResult(False, "p text is empty")

    # 2. Minimum length
    word_count = len(p_text.split())
    if word_count < 50:
        return GateResult(False,
                          f"p text too short ({word_count} words, min 50)")

    # 3. Maximum length — prevents token budget overflow
    if word_count > 1500:
        return GateResult(False,
                          f"p text too long ({word_count} words, max 1500)")

    # 4. Required sections present
    p_lower = p_text.lower()
    for section in ('context:', 'resources:', 'tasks:', 'disruption:'):
        if section not in p_lower:
            return GateResult(False,
                              f"p text missing required section: {section}")

    # 5. At least 2 task references (T + digit)
    task_refs = re.findall(r'\bT\d+\b', p_text, re.IGNORECASE)
    if len(task_refs) < 2:
        return GateResult(False,
                          f"p text has fewer than 2 task references "
                          f"(found {len(task_refs)})")

    # 6. At least one resource reference (handles R1, M2, NR1, Team-A)
    resource_refs = re.findall(
        r'\b(?:R\d+|M\d+|NR\d+)\b|(?:Team-\w+|NR-\w+)',
        p_text, re.IGNORECASE)
    if not resource_refs:
        return GateResult(False, "p text has no resource references")

    # 7. DISRUPTION section has substance
    disrupt_match = re.search(
        r'disruption:(.*?)(?:\n\n|\Z)', p_text, re.DOTALL | re.IGNORECASE)
    if disrupt_match:
        disrupt_words = len(disrupt_match.group(1).split())
        if disrupt_words < 10:
            return GateResult(False,
                              f"DISRUPTION section too short "
                              f"({disrupt_words} words, min 10)")

    # 8. TYPE-B: TASKS section should contain mode notation (T_-M_)
    if ctype == 'TYPE-B':
        if '-M' not in p_text:
            return GateResult(False,
                              "TYPE-B p text has no mode references "
                              "(expected T_-M_ notation in TASKS)")

    return GateResult(True)


# ══════════════════════════════════════════════════════════════════════════════
# GATE 1 — STRUCTURAL VALIDITY (~0ms)
# Checks graph integrity after mutation. Pure Python — no solver, no I/O.
# ══════════════════════════════════════════════════════════════════════════════


def gate_1_structural(mutated_graph: dict) -> GateResult:
    """
    Validates graph structural integrity post-mutation.
    Handles both TYPE-A (flat duration/demands) and
    TYPE-B (modes list) activity schemas.
    """
    ctype = mutated_graph['meta']['type']

    # 1. At least one activity
    if not mutated_graph.get('activities'):
        return GateResult(False, "No activities in graph")

    # 2. All durations positive
    for a in mutated_graph['activities']:
        if ctype in ('TYPE-A', 'TYPE-C'):
            if a.get('duration', 0) <= 0:
                return GateResult(False,
                                  f"Activity {a['id']} has duration "
                                  f"{a.get('duration')} <= 0")
        elif ctype == 'TYPE-B':
            if not a.get('modes'):
                return GateResult(False,
                                  f"Activity {a['id']} has no modes")
            for m in a['modes']:
                if m.get('duration', 0) <= 0:
                    return GateResult(False,
                                      f"Activity {a['id']} mode "
                                      f"{m['mode_id']} has duration "
                                      f"{m.get('duration')} <= 0")

    # 3. All resource capacities positive
    for r in mutated_graph['resources']:
        if r['capacity'] <= 0:
            return GateResult(False,
                              f"Resource {r['id']} has capacity "
                              f"{r['capacity']} <= 0")

    # 4. All demands non-negative
    for a in mutated_graph['activities']:
        if ctype in ('TYPE-A', 'TYPE-C'):
            for rid, dem in a.get('demands', {}).items():
                if dem < 0:
                    return GateResult(False,
                                      f"Activity {a['id']} has negative "
                                      f"demand {dem} for {rid}")
        elif ctype == 'TYPE-B':
            for m in a.get('modes', []):
                for rid, dem in m.get('renewable_demands', {}).items():
                    if dem < 0:
                        return GateResult(
                            False,
                            f"Activity {a['id']} mode {m['mode_id']} "
                            f"has negative renewable demand for {rid}")
                for rid, dem in m.get('nonrenewable_demands', {}).items():
                    if dem < 0:
                        return GateResult(
                            False,
                            f"Activity {a['id']} mode {m['mode_id']} "
                            f"has negative nonrenewable demand for {rid}")

    # 5. Precedence references valid activity IDs
    activity_ids = {a['id'] for a in mutated_graph['activities']}
    for p in mutated_graph['precedences']:
        if p['i'] not in activity_ids:
            return GateResult(False,
                              f"Precedence references unknown source {p['i']}")
        if p['j'] not in activity_ids:
            return GateResult(False,
                              f"Precedence references unknown target {p['j']}")

    # 6. Downtime windows valid
    for r in mutated_graph['resources']:
        for dt in r.get('downtime', []):
            if dt['start'] >= dt['end']:
                return GateResult(False,
                                  f"Resource {r['id']} has downtime "
                                  f"start ({dt['start']}) >= end ({dt['end']})")
            if dt['delta'] <= 0:
                return GateResult(False,
                                  f"Resource {r['id']} has downtime "
                                  f"delta ({dt['delta']}) <= 0")

    # 7. At least one mutation was applied
    if not mutated_graph.get('mutations_applied'):
        return GateResult(False,
                          "No mutations applied — instance identical to base")

    return GateResult(True)


# ══════════════════════════════════════════════════════════════════════════════
# GATE 2 — SYNTAX CHECK (~2ms)
# Safety net for CodeBuilder bugs. Should reject ~0% in production.
# ══════════════════════════════════════════════════════════════════════════════


def gate_2_syntax(code: str) -> GateResult:
    """Runs py_compile on generated code string."""
    fd, path = tempfile.mkstemp(suffix='.py')
    try:
        with os.fdopen(fd, 'w') as f:
            f.write(code)
        py_compile.compile(path, doraise=True)
        return GateResult(True)
    except py_compile.PyCompileError as e:
        return GateResult(False, f"Syntax error: {e}")
    finally:
        if os.path.exists(path):
            os.unlink(path)


# ══════════════════════════════════════════════════════════════════════════════
# GATE 3 — DEDUPLICATION via MinHash LSH (~1ms amortized)
# O(1) near-duplicate detection. Deterministic hashing via hashlib.
# ══════════════════════════════════════════════════════════════════════════════


class MinHashLSH:
    """
    MinHash Locality-Sensitive Hashing for near-duplicate text detection.

    Parameters:
      num_perm:  number of hash permutations (signature size)
      threshold: Jaccard similarity threshold

    Uses hashlib.md5 for deterministic hashing across sessions
    (Python's built-in hash() is randomized per-process since 3.3).

    LSH banding: 16 bands × 8 rows for threshold ≈ 0.85.
    """

    def __init__(self, num_perm: int = 128, threshold: float = 0.85):
        self.num_perm = num_perm
        self.threshold = threshold
        self.num_bands = 16
        self.rows_per_band = num_perm // self.num_bands  # 8

        # Deterministic random coefficients (fixed seed)
        import random as _rng
        rng = _rng.Random(42)
        self.max_hash = (1 << 32) - 1
        self.a = [rng.randint(1, self.max_hash) for _ in range(num_perm)]
        self.b = [rng.randint(0, self.max_hash) for _ in range(num_perm)]
        self.prime = (1 << 61) - 1  # Mersenne prime

        # LSH buckets: band_index → {bucket_hash → [doc_ids]}
        self.buckets: dict[int, dict[int, list[str]]] = {
            i: defaultdict(list) for i in range(self.num_bands)
        }
        self.signatures: dict[str, list[int]] = {}

    def _shingle(self, text: str, k: int = 3) -> set[int]:
        """
        Convert text to word k-shingle hashes.
        Uses hashlib.md5 for deterministic output across
        processes and sessions (immune to PYTHONHASHSEED).
        """
        words = text.lower().split()
        shingles = set()
        for i in range(max(1, len(words) - k + 1)):
            shingle = " ".join(words[i:i + k])
            h = hashlib.md5(shingle.encode('utf-8')).digest()
            shingles.add(int.from_bytes(h[:4], 'little') & self.max_hash)
        return shingles

    def _minhash(self, shingles: set[int]) -> list[int]:
        """Compute MinHash signature from shingle set."""
        sig = []
        for i in range(self.num_perm):
            min_val = float('inf')
            for s in shingles:
                h = (self.a[i] * s + self.b[i]) % self.prime
                min_val = min(min_val, h)
            sig.append(min_val if min_val != float('inf') else 0)
        return sig

    def _band_hashes(self, sig: list[int]) -> list[int]:
        """Split signature into bands and hash each band."""
        bands = []
        for band_idx in range(self.num_bands):
            start = band_idx * self.rows_per_band
            end = start + self.rows_per_band
            # Use hashlib for deterministic band hashing
            payload = json.dumps(sig[start:end]).encode()
            bh = int.from_bytes(
                hashlib.md5(payload).digest()[:8], 'little')
            bands.append(bh)
        return bands

    def is_near_duplicate(self, doc_id: str, text: str) -> bool:
        """
        Check if text is near-duplicate of any registered document.
        O(1) amortized — only verifies candidates in same LSH buckets.
        """
        shingles = self._shingle(text)
        sig = self._minhash(shingles)
        band_hashes = self._band_hashes(sig)

        for band_idx, bh in enumerate(band_hashes):
            if bh in self.buckets[band_idx]:
                candidates = self.buckets[band_idx][bh]
                for cand_id in candidates:
                    cand_sig = self.signatures[cand_id]
                    jaccard = sum(
                        1 for a, b in zip(sig, cand_sig) if a == b
                    ) / self.num_perm
                    if jaccard >= self.threshold:
                        return True
        return False

    def register(self, doc_id: str, text: str):
        """Register document after all gates pass."""
        shingles = self._shingle(text)
        sig = self._minhash(shingles)
        band_hashes = self._band_hashes(sig)
        self.signatures[doc_id] = sig
        for band_idx, bh in enumerate(band_hashes):
            self.buckets[band_idx][bh].append(doc_id)


class DeduplicationRegistry:
    """
    Dedup registry using MinHash LSH + hash dedup + split integrity.
    """

    def __init__(self, sim_threshold: float = 0.85):
        self.seen_hashes: set[str] = set()
        self.lsh = MinHashLSH(num_perm=128, threshold=sim_threshold)
        self.id_to_split: dict[str, str] = {}

    def compute_hash(self, base_id: str,
                     mutations: list[dict]) -> str:
        """Hash of (base_instance_id, mutation_set)."""
        mut_key = json.dumps(
            sorted([
                (m.get('op', ''), m.get('resource', ''),
                 m.get('i', ''), m.get('j', ''),
                 m.get('relation', ''))
                for m in mutations
            ]), sort_keys=True)
        payload = f"{base_id}::{mut_key}"
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def check(self, base_id: str, mutations: list[dict],
              p_text: str, cell_key: str,
              split: str) -> GateResult:
        # 1. Hash dedup — exact (base_id, mutation_set) match
        h = self.compute_hash(base_id, mutations)
        if h in self.seen_hashes:
            return GateResult(False,
                              "Duplicate (base_id, mutation_set) hash")

        # 2. LSH near-duplicate — O(1) amortized
        doc_id = f"{cell_key}::{h}"
        if self.lsh.is_near_duplicate(doc_id, p_text):
            return GateResult(False,
                              "p text is near-duplicate of accepted "
                              "instance (MinHash LSH, Jaccard >= 0.85)")

        # 3. Split integrity — base_id cannot cross splits
        if base_id in self.id_to_split:
            if self.id_to_split[base_id] != split:
                return GateResult(False,
                                  f"Split leak: base_id '{base_id}' in "
                                  f"'{self.id_to_split[base_id]}' but "
                                  f"candidate is '{split}'")

        return GateResult(True)

    def register(self, base_id: str, mutations: list[dict],
                 p_text: str, cell_key: str, split: str):
        h = self.compute_hash(base_id, mutations)
        self.seen_hashes.add(h)
        doc_id = f"{cell_key}::{h}"
        self.lsh.register(doc_id, p_text)
        self.id_to_split[base_id] = split


# ══════════════════════════════════════════════════════════════════════════════
# SOLVER POOL — Persistent worker pool using exec()
# Cross-platform timeout via threading (no signal.SIGALRM).
# Future-based result routing (no put-back loop).
# ══════════════════════════════════════════════════════════════════════════════


def _solver_worker(task_queue: mp.Queue,
                   result_dict: dict,
                   result_lock: mp.Lock):
    """
    Worker process. Executes code via exec() — no subprocess cold start.
    OR-Tools imported once at process startup.
    Cross-platform timeout via threading.Thread.join(timeout=N).
    """
    # Pre-import OR-Tools once per worker lifetime
    try:
        from ortools.sat.python import cp_model  # noqa: F401
    except ImportError:
        pass  # will fail at exec time with clear error

    while True:
        task = task_queue.get()
        if task is None:
            break  # poison pill — shutdown

        task_id, code_str, timeout_sec = task
        result = {'success': False, 'error': 'Unknown error'}

        def _exec_code():
            """Runs exec() in a thread for timeout control."""
            nonlocal result
            stdout_capture = io.StringIO()
            try:
                exec_globals = {'__builtins__': __builtins__}
                with redirect_stdout(stdout_capture):
                    exec(code_str, exec_globals, {})

                output_str = stdout_capture.getvalue().strip()
                if output_str:
                    try:
                        output = json.loads(output_str)
                        result = {'success': True, 'output': output}
                    except json.JSONDecodeError:
                        result = {'success': False,
                                  'error': f'Bad JSON: {output_str[:200]}'}
                else:
                    result = {'success': False,
                              'error': 'No output produced'}

            except Exception as e:
                result = {'success': False,
                          'error': f'{type(e).__name__}: {str(e)[:300]}'}

        # Run exec in a daemon thread with timeout
        t = threading.Thread(target=_exec_code, daemon=True)
        t.start()
        t.join(timeout=timeout_sec)

        if t.is_alive():
            # Thread still running — timed out
            # Thread is daemon — will die when worker process exits
            result = {'success': False,
                      'error': f'Timeout after {timeout_sec}s'}

        # Route result to caller via shared dict
        with result_lock:
            result_dict[task_id] = result


class SolverPool:
    """
    Pool of persistent solver workers.
    exec() in worker — no subprocess cold start.
    Per-task result routing via shared dict — no put-back loop.
    Thread-safe for concurrent callers.

    Usage:
        pool = SolverPool(n_workers=4)
        result = pool.solve(code_string, timeout=10)
        results = pool.solve_batch([(code1, 10), (code2, 90)])
        pool.shutdown()
    """

    def __init__(self, n_workers: int = 4):
        self._manager = mp.Manager()
        self._task_queue = mp.Queue()
        self._result_dict = self._manager.dict()
        self._result_lock = mp.Lock()
        self._task_counter = 0
        self._counter_lock = threading.Lock()
        self._workers: list[mp.Process] = []

        for _ in range(n_workers):
            w = mp.Process(
                target=_solver_worker,
                args=(self._task_queue,
                      self._result_dict,
                      self._result_lock),
                daemon=True)
            w.start()
            self._workers.append(w)

    def _next_task_id(self) -> int:
        with self._counter_lock:
            tid = self._task_counter
            self._task_counter += 1
            return tid

    def solve(self, code: str, timeout: int = 10) -> dict:
        """
        Submit code for execution. Blocking — returns result.
        Polls its own slot in shared dict. No put-back loop.
        """
        task_id = self._next_task_id()
        self._task_queue.put((task_id, code, timeout))

        deadline = time.time() + timeout + 15
        while time.time() < deadline:
            with self._result_lock:
                if task_id in self._result_dict:
                    return self._result_dict.pop(task_id)
            time.sleep(0.05)  # 50ms poll interval

        return {'success': False,
                'error': f'Pool timeout: no result for task '
                         f'{task_id} after {timeout + 15}s'}

    def solve_batch(self, tasks: list[tuple[str, int]]) -> list[dict]:
        """
        Submit multiple (code, timeout) pairs for parallel execution.
        Returns results in same order as input.
        """
        task_ids = []
        for code, timeout in tasks:
            tid = self._next_task_id()
            task_ids.append(tid)
            self._task_queue.put((tid, code, timeout))

        results: list[Optional[dict]] = [None] * len(task_ids)
        max_timeout = max(t for _, t in tasks) + 15
        deadline = time.time() + max_timeout

        collected = 0
        while collected < len(task_ids) and time.time() < deadline:
            with self._result_lock:
                for i, tid in enumerate(task_ids):
                    if results[i] is None and tid in self._result_dict:
                        results[i] = self._result_dict.pop(tid)
                        collected += 1
            if collected < len(task_ids):
                time.sleep(0.05)

        # Fill missing with timeout errors
        for i in range(len(results)):
            if results[i] is None:
                results[i] = {'success': False, 'error': 'Pool timeout'}
        return results

    def check_workers_alive(self):
        """Check and restart any dead workers."""
        for i, w in enumerate(self._workers):
            if not w.is_alive():
                new_w = mp.Process(
                    target=_solver_worker,
                    args=(self._task_queue,
                          self._result_dict,
                          self._result_lock),
                    daemon=True)
                new_w.start()
                self._workers[i] = new_w

    def shutdown(self):
        for _ in self._workers:
            self._task_queue.put(None)
        for w in self._workers:
            w.join(timeout=5)
        try:
            self._manager.shutdown()
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════════════════════
# GATE 4 — CP-SAT FEASIBILITY (~10s via SolverPool)
# ══════════════════════════════════════════════════════════════════════════════


def gate_4_feasibility(code: str,
                       solver_pool: SolverPool,
                       timeout: int = 10) -> GateResult:
    """
    Executes generated code with short time limit.
    Checks CP-SAT finds at least one feasible solution.
    Uses SolverPool (exec in worker) — no subprocess cold start.
    """
    from code_builder import set_solver_timeout
    patched = set_solver_timeout(code, timeout)

    result = solver_pool.solve(patched, timeout=timeout + 5)

    if not result['success']:
        return GateResult(False,
                          f"Feasibility check failed: {result['error']}")

    output = result['output']
    status = output.get('status', 'unknown')
    if status in ('optimal', 'feasible'):
        return GateResult(True,
                          metadata={'status': status,
                                    'objective': output.get('objective')})
    return GateResult(False,
                      f"CP-SAT status: {status} "
                      f"(no feasible solution within {timeout}s)")


# ══════════════════════════════════════════════════════════════════════════════
# GATE 5 — DISTRIBUTION QUOTA (~0ms)
# ══════════════════════════════════════════════════════════════════════════════


class DistributionQuota:
    """
    Tracks counts per (type, difficulty, qtype) cell.
    Holds excess rather than discarding.
    Target: 30% easy / 50% medium / 20% hard per type.
    """

    def __init__(self, targets_per_type: int = 15000):
        self.targets = targets_per_type
        self.difficulty_ratios = {
            'easy': 0.30, 'medium': 0.50, 'hard': 0.20
        }
        self.counts: dict[str, int] = defaultdict(int)

    def cell_key(self, ctype: str, difficulty: str,
                 qtype: str) -> str:
        return f"{ctype}::{difficulty}::{qtype}"

    def max_for_difficulty(self, difficulty: str) -> int:
        ratio = self.difficulty_ratios.get(difficulty, 0.50)
        return int(self.targets * ratio)

    def check(self, ctype: str, difficulty: str,
              qtype: str) -> GateResult:
        key = self.cell_key(ctype, difficulty, qtype)
        count = self.counts[key]
        max_count = self.max_for_difficulty(difficulty)

        if count >= max_count:
            return GateResult(False,
                              f"Quota full for cell {key}: "
                              f"{count} >= {max_count}",
                              metadata={'action': 'hold', 'count': count})
        return GateResult(True, metadata={'count': count})

    def increment(self, ctype: str, difficulty: str, qtype: str):
        key = self.cell_key(ctype, difficulty, qtype)
        self.counts[key] += 1


# ══════════════════════════════════════════════════════════════════════════════
# GATE 6 — p↔m CONSISTENCY (~5ms)
# Keyword mapper with word-level proximity, sentence-scoped time lag,
# dynamic resource ID patterns (hyphen-safe), relation-change detection.
# ══════════════════════════════════════════════════════════════════════════════

# ── Keyword maps ──────────────────────────────────────────────────────────────

DOWNTIME_PHRASES = [
    'offline', 'unavailable', 'shutdown', 'down',
    'capacity reduced', 'capacity cut', 'failure',
    'malfunction', 'breakdown', 'out of service',
    'contamination', 'maintenance', 'repair',
    'reduced by', 'drops to', 'limited to',
    'coolant failure', 'power outage', 'blocked',
]

TIME_LAG_PHRASES_FS = [
    r'must wait', r'cannot start until', r'after.*finish',
    r'minimum.*gap', r'\blag\b', r'\bdelay\b',
    r'cool(?:ing)?', r'cur(?:e|ing)', r'dry(?:ing)?',
    r'decontaminat', r'steriliz', r'separation',
    r'must not start before', r'at least.*hour',
    r'\bbuffer\b', r'mandatory gap', r'waiting period',
    r'must start within', r'no later than',
    r'time window', r'within.*hour', r'cannot be delayed',
]

TIME_LAG_PHRASES_SS = [
    r'start.*together', r'simultaneously', r'at the same time',
    r'concurrent', r'must begin.*when', r'same.*start',
    r'coordinated start',
]

TIME_LAG_PHRASES_FF = [
    r'finish.*together', r'complete.*simultaneously',
    r'must end.*before.*end', r'finish.*within.*of.*finish',
    r'completion.*window',
]

OBJECTIVE_PHRASES = {
    'minimize_weighted_tardiness': [
        'penalty', 'deadline', 'late', 'overdue',
        'minimize.*delay', 'due date', 'tardiness',
        'priority', 'weighted', 'urgent',
    ],
    'minimize_resource_cost': [
        'cost', 'budget', 'expense', 'minimize.*resource',
        'resource.*cost', 'cheaper', 'economical',
    ],
    'maximize_robustness_margin': [
        'robust', 'buffer', 'slack', 'safety margin',
        'contingency', 'risk', 'uncertainty',
    ],
}

MODE_RESTRICTION_PHRASES = [
    'unavailable', 'not available', 'restricted to',
    'cannot use', 'forced to', 'only option',
    'engineer.*left', 'resource.*shortage',
    'mode.*removed', 'option.*removed',
    'slow.*mode', 'fast.*mode.*unavailable',
    'team.*reduced', 'contractor.*unavailable',
]


# ── Helper: word-level proximity regex ────────────────────────────────────────

def _word_proximity_pattern(term_a: str, term_b: str,
                            max_words: int) -> str:
    """
    Regex matching term_a and term_b with at most max_words
    intervening words (either order). Counts word tokens, not chars.
    """
    gap = rf'(?:\S+\s+){{0,{max_words}}}\s*'
    a = re.escape(term_a)
    b = re.escape(term_b)
    return rf'(?:{a}\s+{gap}{b}|{b}\s+{gap}{a})'


# ── Helper: build resource ID pattern (hyphen-safe) ───────────────────────────

def _build_resource_id_pattern(m: dict) -> tuple[str, list[str]]:
    """
    Builds regex matching any resource ID from target m.
    Handles R1, M2, Team-A, NR-Budget (hyphen-safe boundaries).
    """
    resource_ids = [r['id'] for r in m['resources']]
    patterns = []
    for rid in resource_ids:
        escaped = re.escape(rid)
        pat = rf'(?<![A-Za-z0-9_]){escaped}(?![A-Za-z0-9_])'
        patterns.append(pat)
    combined = '(' + '|'.join(patterns) + ')' if patterns else '(?!)'
    return combined, resource_ids


# ── Signal extraction from target m ───────────────────────────────────────────

@dataclass
class ConstraintSignal:
    type: str
    resource: Optional[str] = None
    activity_i: Optional[int] = None
    activity_j: Optional[int] = None
    relation: Optional[str] = None
    base_relation: Optional[str] = None
    obj_type: Optional[str] = None
    activity_id: Optional[int] = None


def _extract_m_signals(target_m: dict,
                       base_graph: dict) -> list[ConstraintSignal]:
    """
    Reads target m and base_graph. Produces signals that MUST
    have linguistic evidence in p's DISRUPTION section.
    Handles additive precedence model (multiple relations per pair).
    """
    signals = []

    # 1. Resource downtime
    for r in target_m['resources']:
        for dt in r.get('downtime', []):
            signals.append(ConstraintSignal(
                type='downtime', resource=r['id']))

    # 2. Precedences: detect new edges, tightened lags, relation changes
    base_precs = {}
    for p in base_graph.get('precedences', []):
        key = (p['i'], p['j'], p['relation'])
        base_precs[key] = p

    for p in target_m['precedences']:
        key = (p['i'], p['j'], p['relation'])
        base_p = base_precs.get(key)

        is_new_edge = base_p is None
        lag_min_increased = (base_p is not None and
                             p['lag_min'] > base_p.get('lag_min', 0))
        lag_max_added = (p.get('lag_max') is not None and
                         (base_p is None or
                          base_p.get('lag_max') is None))
        # Relation change: check if same (i,j) pair has a DIFFERENT
        # relation type that didn't exist in base
        pair_key = (p['i'], p['j'])
        pair_existed_with_this_rel = any(
            bp['relation'] == p['relation']
            for bp_key, bp in base_precs.items()
            if bp_key[:2] == pair_key)
        relation_is_new = not pair_existed_with_this_rel

        if any([is_new_edge, lag_min_increased,
                lag_max_added, relation_is_new]):
            signals.append(ConstraintSignal(
                type='time_lag',
                activity_i=p['i'], activity_j=p['j'],
                relation=p['relation']))

    # 3. Objective change
    if target_m['objective']['type'] != 'minimize_makespan':
        signals.append(ConstraintSignal(
            type='objective_change',
            obj_type=target_m['objective']['type']))

    # 4. TYPE-B mode restrictions
    base_mode_counts = {}
    for a in base_graph.get('activities', []):
        if 'modes' in a:
            base_mode_counts[a['id']] = len(a['modes'])

    for a in target_m.get('activities', []):
        if 'modes' in a and a['id'] in base_mode_counts:
            if len(a['modes']) < base_mode_counts[a['id']]:
                signals.append(ConstraintSignal(
                    type='mode_restriction',
                    activity_id=a['id']))

    return signals


# ── Direction A: m → p (every m constraint described in p) ────────────────────

PROXIMITY_WORDS = 10


def _check_direction_a(disruption_text: str,
                       target_m: dict,
                       base_graph: dict) -> GateResult:
    """Every constraint signal in m has linguistic evidence in p."""
    signals = _extract_m_signals(target_m, base_graph)
    sentences = re.split(r'(?<=[.!?])\s+', disruption_text.lower())

    for sig in signals:
        found = False

        if sig.type == 'downtime':
            rid = sig.resource.lower()
            for phrase in DOWNTIME_PHRASES:
                pattern = _word_proximity_pattern(
                    rid, phrase, PROXIMITY_WORDS)
                if re.search(pattern, disruption_text.lower(),
                             re.IGNORECASE):
                    found = True
                    break
            if not found:
                return GateResult(False,
                                  f"Downtime for {sig.resource} in m "
                                  f"has no signal in p DISRUPTION")

        elif sig.type == 'time_lag':
            ti = f't{sig.activity_i}'
            tj = f't{sig.activity_j}'
            rel = sig.relation or 'FS'
            if rel in ('FS', 'SF'):
                phrases = TIME_LAG_PHRASES_FS
            elif rel == 'SS':
                phrases = TIME_LAG_PHRASES_SS
            elif rel == 'FF':
                phrases = TIME_LAG_PHRASES_FF
            else:
                phrases = TIME_LAG_PHRASES_FS

            # Sentence-scoped: both task IDs + phrase in same sentence
            for sentence in sentences:
                if ti not in sentence or tj not in sentence:
                    continue
                for phrase in phrases:
                    if re.search(phrase, sentence):
                        found = True
                        break
                if found:
                    break
            if not found:
                return GateResult(
                    False,
                    f"Time lag T{sig.activity_i}→T{sig.activity_j} "
                    f"({rel}) in m has no signal in p DISRUPTION")

        elif sig.type == 'objective_change':
            obj_type = sig.obj_type
            phrases = OBJECTIVE_PHRASES.get(obj_type, [])
            for phrase in phrases:
                if phrase in disruption_text.lower():
                    found = True
                    break
            if not found:
                return GateResult(
                    False,
                    f"Objective '{obj_type}' in m has no signal in p")

        elif sig.type == 'mode_restriction':
            aid = str(sig.activity_id)
            for phrase in MODE_RESTRICTION_PHRASES:
                for sentence in sentences:
                    if f't{aid}' in sentence and re.search(phrase, sentence):
                        found = True
                        break
                if found:
                    break
            if not found:
                return GateResult(
                    False,
                    f"Mode restriction on T{sig.activity_id} "
                    f"has no signal in p DISRUPTION")

    return GateResult(True)


# ── Direction B: p → m (p does not describe constraints absent from m) ────────

def _check_direction_b(disruption_text: str,
                       target_m: dict) -> GateResult:
    """
    Checks p does not describe downtime for resources
    with no downtime in m. Uses dynamic resource ID pattern
    (hyphen-safe) and sentence-scoped matching.
    """
    resource_pattern, all_ids = _build_resource_id_pattern(target_m)
    m_resources_with_downtime = {
        r['id'].lower() for r in target_m['resources']
        if r.get('downtime')
    }
    sentences = re.split(r'(?<=[.!?])\s+', disruption_text.lower())

    for sentence in sentences:
        mentioned = re.findall(resource_pattern, sentence, re.IGNORECASE)
        mentioned = [rid.lower() for rid in mentioned if rid]
        if not mentioned:
            continue
        has_downtime_phrase = any(
            re.search(re.escape(phrase), sentence)
            for phrase in DOWNTIME_PHRASES)
        if not has_downtime_phrase:
            continue
        for rid in mentioned:
            if rid not in m_resources_with_downtime:
                return GateResult(
                    False,
                    f"p describes downtime for '{rid}' but "
                    f"m has no downtime for this resource "
                    f"(hallucinated constraint)")

    return GateResult(True)


def gate_6_consistency(p_text: str, target_m: dict,
                       base_graph: dict) -> GateResult:
    """
    Gate 6: bidirectional p↔m consistency check.
    Direction A: every m constraint has signal in p.
    Direction B: p does not hallucinate constraints absent from m.
    """
    # Extract DISRUPTION section only
    disrupt_match = re.search(
        r'disruption:(.*?)(?:\n\n|\Z)', p_text,
        re.DOTALL | re.IGNORECASE)
    if not disrupt_match:
        return GateResult(False, "p missing DISRUPTION section")
    disruption_text = disrupt_match.group(1)

    result_a = _check_direction_a(disruption_text, target_m, base_graph)
    if not result_a.passed:
        return GateResult(False, f"Direction A (m→p): {result_a.reason}")

    result_b = _check_direction_b(disruption_text, target_m)
    if not result_b.passed:
        return GateResult(False, f"Direction B (p→m): {result_b.reason}")

    return GateResult(True)


# ══════════════════════════════════════════════════════════════════════════════
# GATE 7 — SOLUTION QUALITY (60–90s via SolverPool)
# Gap thresholds per difficulty. Continuous gap-weighted loss.
# BestObjectiveBound guarded against extreme values.
# ══════════════════════════════════════════════════════════════════════════════


def gate_7_quality(code: str, difficulty: str,
                   solver_pool: SolverPool,
                   timeout: int = 90) -> GateResult:
    """
    Full oracle solve with gap analysis.
    Thresholds: easy ≤2%, medium ≤5%, hard ≤15%.
    approx_flag set when feasible but gap > threshold.
    BestObjectiveBound guarded against extreme values (±1e15).
    """
    GAP_THRESHOLDS = {
        'easy': 0.02, 'medium': 0.05, 'hard': 0.15
    }

    from code_builder import set_solver_timeout, enable_bound_output
    patched = set_solver_timeout(code, timeout)
    patched = enable_bound_output(patched)

    result = solver_pool.solve(patched, timeout=timeout + 10)

    if not result['success']:
        return GateResult(False,
                          f"Oracle solve failed: {result['error']}")

    output = result['output']
    status = output.get('status', 'unknown')

    if status == 'infeasible':
        return GateResult(False,
                          "Oracle infeasible (Gate 4 false positive?)")

    obj = output.get('objective', 0)
    bound = output.get('best_bound')

    # Guard: extreme best_bound → treat as gap unknown
    if bound is None or abs(bound) > 1e15:
        bound = obj

    # Compute optimality gap
    if obj == 0:
        gap = 0.0
    else:
        gap = abs(obj - bound) / max(abs(obj), 1e-9)

    threshold = GAP_THRESHOLDS.get(difficulty, 0.05)
    approx = False

    if status == 'optimal':
        gap = 0.0
    elif gap > threshold and status == 'feasible':
        approx = True

    return GateResult(True,
                      metadata={
                          'status': status,
                          'objective': obj,
                          'best_bound': bound,
                          'gap': round(gap, 6),
                          'threshold': threshold,
                          'approx_flag': approx
                      })


# ══════════════════════════════════════════════════════════════════════════════
# LOSS WEIGHT COMPUTATION
# Continuous weighting based on optimality gap.
# ══════════════════════════════════════════════════════════════════════════════


def _compute_loss_weight(gap: float, approx: bool) -> float:
    """
    Continuous loss weight for SFT training.
    gap=0.0  (proven optimal)  → weight=1.0
    gap=0.05 (5% gap)          → weight≈0.83
    gap=0.15 (15% gap)         → weight=0.5
    approx_flag=True           → weight=0.3 (hard cap)
    """
    if approx:
        return 0.3
    weight = 1.0 - (gap / 0.15) * 0.5
    return max(0.5, min(1.0, weight))


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING TRIPLET BUILDER
# Assembles final (p, m, c) triplet with quality metadata.
# ══════════════════════════════════════════════════════════════════════════════


def build_training_triplet(p_text: str, target_m: dict,
                           code: str,
                           pipeline_result: PipelineResult,
                           graph: dict) -> dict:
    """
    Assembles final training triplet with loss weight metadata.
    Called only after FilterPipeline.run() returns accepted=True.
    """
    g7_meta = pipeline_result.gate_results.get(
        7, GateResult(True, metadata={}))

    gap = g7_meta.metadata.get('gap', 0.0)
    approx = pipeline_result.approx_flag

    return {
        # Training content (what the model sees)
        "p": p_text,
        "m": target_m,
        "c": code,

        # Quality metadata
        "approx_flag": approx,
        "optimality_gap": gap,
        "oracle_status": g7_meta.metadata.get('status', 'unknown'),
        "oracle_objective": g7_meta.metadata.get('objective'),
        "loss_weight": _compute_loss_weight(gap, approx),

        # Pipeline metadata (not seen by model)
        "base_instance_id": graph['meta']['instance_id'],
        "constraint_type": graph['meta']['type'],
        "difficulty": graph['meta']['difficulty'],
        "question_type": graph['meta'].get('question_type'),
        "split": graph['meta']['split'],
    }


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE ORCHESTRATOR
# Runs all 8 gates (0–7) in cost-first order. Short-circuits on failure.
# ══════════════════════════════════════════════════════════════════════════════


class FilterPipeline:
    """
    Runs 8 filtering gates in cost-first order.
    Maintains stateful registries for dedup and quota.
    Uses SolverPool for Gates 4 and 7.
    """

    def __init__(self, targets_per_type: int = 15000,
                 n_solver_workers: int = 4):
        self.dedup_registry = DeduplicationRegistry()
        self.quota = DistributionQuota(targets_per_type)
        self.solver_pool = SolverPool(n_workers=n_solver_workers)
        self.stats: dict[str, int] = defaultdict(int)

    def run(self, p_text: str, mutated_graph: dict,
            code: str, target_m: dict,
            base_graph: dict) -> PipelineResult:
        """
        Runs all gates on a candidate triplet.
        Short-circuits on first failure.
        Returns PipelineResult with per-gate diagnostics.
        """
        meta = mutated_graph['meta']
        base_id = meta['instance_id']
        ctype = meta['type']
        difficulty = meta['difficulty']
        qtype = meta.get('question_type', 'optimization')
        split = meta['split']
        mutations = mutated_graph.get('mutations_applied', [])
        cell_key = self.quota.cell_key(ctype, difficulty, qtype)

        gate_results = {}

        # ── Gate 0: Semantic pre-filter (~0.1ms) ──────────────
        g0 = gate_0_semantic_prefilter(p_text, ctype)
        gate_results[0] = g0
        if not g0.passed:
            self.stats['gate_0_reject'] += 1
            return PipelineResult(accepted=False, failed_gate=0,
                                  gate_results=gate_results)

        # ── Gate 1: Structural validity (~0ms) ────────────────
        g1 = gate_1_structural(mutated_graph)
        gate_results[1] = g1
        if not g1.passed:
            self.stats['gate_1_reject'] += 1
            return PipelineResult(accepted=False, failed_gate=1,
                                  gate_results=gate_results)

        # ── Gate 2: Syntax check (~2ms) ───────────────────────
        g2 = gate_2_syntax(code)
        gate_results[2] = g2
        if not g2.passed:
            self.stats['gate_2_reject'] += 1
            return PipelineResult(accepted=False, failed_gate=2,
                                  gate_results=gate_results)

        # ── Gate 3: Deduplication (~1ms) ──────────────────────
        g3 = self.dedup_registry.check(
            base_id, mutations, p_text, cell_key, split)
        gate_results[3] = g3
        if not g3.passed:
            self.stats['gate_3_reject'] += 1
            return PipelineResult(accepted=False, failed_gate=3,
                                  gate_results=gate_results)

        # ── Gate 4: CP-SAT feasibility (~10s) ─────────────────
        self.solver_pool.check_workers_alive()
        g4 = gate_4_feasibility(code, self.solver_pool, timeout=10)
        gate_results[4] = g4
        if not g4.passed:
            self.stats['gate_4_reject'] += 1
            return PipelineResult(accepted=False, failed_gate=4,
                                  gate_results=gate_results)

        # ── Gate 5: Distribution quota (~0ms) ─────────────────
        g5 = self.quota.check(ctype, difficulty, qtype)
        gate_results[5] = g5
        if not g5.passed:
            self.stats['gate_5_hold'] += 1
            return PipelineResult(accepted=False, failed_gate=5,
                                  gate_results=gate_results)

        # ── Gate 6: p↔m consistency (~5ms) ────────────────────
        g6 = gate_6_consistency(p_text, target_m, base_graph)
        gate_results[6] = g6
        if not g6.passed:
            self.stats['gate_6_reject'] += 1
            return PipelineResult(accepted=False, failed_gate=6,
                                  gate_results=gate_results)

        # ── Gate 7: Solution quality (60–90s) ─────────────────
        g7 = gate_7_quality(code, difficulty, self.solver_pool,
                            timeout=90)
        gate_results[7] = g7
        if not g7.passed:
            self.stats['gate_7_reject'] += 1
            return PipelineResult(accepted=False, failed_gate=7,
                                  gate_results=gate_results)

        # ── All gates passed — register and accept ────────────
        approx = g7.metadata.get('approx_flag', False)

        self.dedup_registry.register(
            base_id, mutations, p_text, cell_key, split)
        self.quota.increment(ctype, difficulty, qtype)
        self.stats['accepted'] += 1
        if approx:
            self.stats['accepted_approx'] += 1

        return PipelineResult(
            accepted=True,
            gate_results=gate_results,
            approx_flag=approx)

    def report(self) -> dict:
        """Returns summary statistics for logging and diagnostics."""
        total_processed = sum(self.stats.values())
        accepted = self.stats.get('accepted', 0)
        return {
            'total_candidates': total_processed,
            'accepted': accepted,
            'accepted_approx': self.stats.get('accepted_approx', 0),
            'acceptance_rate': accepted / max(1, total_processed),
            'rejections_by_gate': {
                k: v for k, v in sorted(self.stats.items())
                if 'reject' in k or 'hold' in k
            },
            'quota_distribution': dict(self.quota.counts),
        }

    def shutdown(self):
        """Clean up solver pool."""
        self.solver_pool.shutdown()



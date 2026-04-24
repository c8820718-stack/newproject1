"""
gpt4o_contextualizer.py — GPT-4o Two-Step p Generation
=======================================================
Step B1: graph summary + 5-slot prompt → mutation spec JSON (severity only)
Step B2: pre-written RESOURCES+TASKS + resolved mutations → CONTEXT + DISRUPTION

Design constraints (from all review rounds):
  - GPT-4o NEVER sees full graph JSON — only the ~100-token summary header
  - GPT-4o NEVER writes task durations, demands, or precedences — only narrative
  - GPT-4o outputs severity levels, not exact numbers (no numerical hallucination)
  - DISRUPTION text uses exact resolved values (Step B2 runs AFTER GraphMutator)
  - Retry logic with exponential backoff on malformed JSON output
  - Cost tracking per call for budget estimation

Depends on: instance_parser.py (build_graph_summary)
"""

import json
import time
import re
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class ContextualizerResult:
    success: bool
    mutation_spec: Optional[list[dict]] = None
    narrative_context: Optional[str] = None
    narrative_disruption: Optional[str] = None
    error: Optional[str] = None
    attempts: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0


@dataclass
class CostTracker:
    """Tracks cumulative GPT-4o API costs across all calls."""
    total_calls: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    failed_calls: int = 0
    retries: int = 0

    # GPT-4o pricing (as of early 2026 — verify before running)
    INPUT_COST_PER_1K: float = 0.0025   # $2.50 per 1M input tokens
    OUTPUT_COST_PER_1K: float = 0.01    # $10.00 per 1M output tokens

    def record(self, input_tokens: int, output_tokens: int,
               success: bool):
        self.total_calls += 1
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cost_usd += (
            input_tokens / 1000 * self.INPUT_COST_PER_1K +
            output_tokens / 1000 * self.OUTPUT_COST_PER_1K
        )
        if not success:
            self.failed_calls += 1

    def report(self) -> dict:
        return {
            'total_calls': self.total_calls,
            'failed_calls': self.failed_calls,
            'retries': self.retries,
            'total_input_tokens': self.total_input_tokens,
            'total_output_tokens': self.total_output_tokens,
            'total_cost_usd': round(self.total_cost_usd, 4),
            'avg_cost_per_call': round(
                self.total_cost_usd / max(1, self.total_calls), 6),
        }


# ══════════════════════════════════════════════════════════════════════════════
# ALLOWED OPS PER TYPE (same as validator — single source of truth)
# ══════════════════════════════════════════════════════════════════════════════

ALLOWED_OPS = {
    "TYPE-A": [
        "reduce_capacity", "freeze_resource",
        "enforce_time_lag", "add_precedence", "alter_objective",
    ],
    "TYPE-B": [
        "alter_mode_profile", "restrict_modes",
        "add_mode", "reduce_nonrenewable_budget",
        "reduce_capacity", "alter_objective",
    ],
    "TYPE-C": [
        "reduce_capacity", "freeze_resource",
        "enforce_time_lag", "add_precedence", "alter_objective",
    ],
}

MUTATION_BUDGETS = {
    'easy':   (1, 2),
    'medium': (2, 4),
    'hard':   (3, 5),
}


# ══════════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPTS
# ══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT_B1 = """You are a scheduling crisis generator for an operations research data pipeline.
Your outputs feed a deterministic Python engine (GraphMutator) — you never write code yourself.

YOUR TWO OUTPUTS per call:
1. narrative_sketch: a 1-2 sentence description of the crisis scenario (used later for full narrative)
2. mutations: a JSON array of mutation specs

STRICT RULES:
- Use ONLY task IDs and resource IDs listed in [GRAPH SUMMARY]
- Use ONLY ops listed in [ALLOWED_OPS]
- Use ONLY severity levels: "minor" | "moderate" | "major" | "critical"
- For time lag mutations: specify relation as "FS" | "SS" | "FF" | "SF"
- If specifying a maximum lag: add "has_max_lag": true and "max_lag_severity": one of the four levels
- For positional parameters (t_start): use "early" | "mid" | "late"
- Do NOT invent task IDs or resource IDs not in the summary
- Do NOT specify exact numerical values — use severity levels ONLY
- Output ONLY valid JSON with exactly this structure:
  {"narrative_sketch": "...", "mutations": [...]}
"""

SYSTEM_PROMPT_B2 = """You write ONLY the CONTEXT and DISRUPTION sections of a scheduling crisis description.
You receive pre-written RESOURCES and TASKS sections based on real data — do NOT modify them.

CONTEXT: 2-4 sentences setting the industrial/domain scene. Include facility type, team size, and what triggered the crisis.
DISRUPTION: Describe EACH disruption listed in [RESOLVED_MUTATIONS] using natural language.
All numerical values are provided in [RESOLVED_MUTATIONS] — use them exactly as given.

STRICT RULES:
- Do NOT invent any numbers, constraints, or disruptions not in [RESOLVED_MUTATIONS]
- Do NOT repeat the RESOURCES or TASKS sections
- Do NOT write any code or mathematical formulas
- Output plain text with exactly two sections: CONTEXT: ... and DISRUPTION: ...
"""


# ══════════════════════════════════════════════════════════════════════════════
# FEW-SHOT EXAMPLES (one per type)
# ══════════════════════════════════════════════════════════════════════════════

FEW_SHOT_B1 = {
    "TYPE-A": {
        "narrative_sketch": "A coolant system failure in Zone B caused partial machine shutdown and contamination requiring decontamination protocols.",
        "mutations": [
            {"op": "reduce_capacity", "resource": "R2",
             "severity": "major", "t_start": "early", "t_end": "mid"},
            {"op": "enforce_time_lag", "i": 7, "j": 9,
             "relation": "FS", "lag_severity": "moderate",
             "has_max_lag": False},
        ]
    },
    "TYPE-B": {
        "narrative_sketch": "A senior engineer departure mid-sprint forces the team to use slower execution modes and reduces the available budget.",
        "mutations": [
            {"op": "restrict_modes", "activity_id": 3,
             "remove_mode_ids": [1], "reason": "resource_unavailability"},
            {"op": "reduce_nonrenewable_budget", "resource": "NR1",
             "severity": "moderate"},
        ]
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# DOMAIN DESCRIPTIONS (for CONTEXT variety)
# ══════════════════════════════════════════════════════════════════════════════

DOMAIN_DESCRIPTIONS = {
    "TYPE-A": [
        "semiconductor fabrication line",
        "chemical processing plant",
        "automotive assembly line",
        "food processing facility",
        "pharmaceutical production line",
        "steel manufacturing plant",
        "electronics assembly factory",
        "textile production facility",
    ],
    "TYPE-B": [
        "software development sprint",
        "consulting engagement project",
        "IT infrastructure migration",
        "product development cycle",
        "research and development project",
        "system integration project",
        "marketing campaign rollout",
        "data center deployment",
    ],
}


# ══════════════════════════════════════════════════════════════════════════════
# GPT-4o CONTEXTUALIZER
# ══════════════════════════════════════════════════════════════════════════════


class GPT4oContextualizer:
    """
    Two-step GPT-4o pipeline for generating p text and mutation specs.

    Step B1: graph summary → mutation spec JSON (severity levels only)
    Step B2: pre-written structure + resolved mutations → CONTEXT + DISRUPTION

    Requires: openai package installed, OPENAI_API_KEY environment variable set.
    """

    def __init__(self, model: str = "gpt-4o",
                 max_retries: int = 3,
                 retry_delay_base: float = 2.0,
                 temperature: float = 0.7):
        self.model = model
        self.max_retries = max_retries
        self.retry_delay_base = retry_delay_base
        self.temperature = temperature
        self.cost_tracker = CostTracker()

        # Lazy import — only needed when actually calling API
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI()
            except ImportError:
                raise ImportError(
                    "openai package not installed. "
                    "Run: pip install openai")
        return self._client

    # ── Step B1: Mutation spec generation ─────────────────────────────────

    def generate_mutation_spec(
            self, graph_summary: str, ctype: str,
            difficulty: str, question_type: str,
            domain_hint: Optional[str] = None
    ) -> ContextualizerResult:
        """
        Step B1: graph summary → mutation spec JSON.
        GPT-4o outputs severity levels and op names — no exact numbers.
        Retries up to max_retries times on malformed JSON.
        """
        import random

        allowed_ops = ALLOWED_OPS.get(ctype, ALLOWED_OPS["TYPE-A"])
        min_ops, max_ops = MUTATION_BUDGETS.get(difficulty, (2, 4))

        if domain_hint is None:
            domains = DOMAIN_DESCRIPTIONS.get(ctype,
                                              DOMAIN_DESCRIPTIONS["TYPE-A"])
            domain_hint = random.choice(domains)

        few_shot = FEW_SHOT_B1.get(ctype, FEW_SHOT_B1["TYPE-A"])

        user_prompt = (
            f"[TYPE: {ctype}] [DIFF: {difficulty}] [QTYPE: {question_type}]\n\n"
            f"[GRAPH SUMMARY]\n{graph_summary}\n[END SUMMARY]\n\n"
            f"[ALLOWED_OPS: {', '.join(allowed_ops)}]\n"
            f"[MUTATION_BUDGET: {min_ops}-{max_ops} ops]\n\n"
            f"[FEW_SHOT_EXAMPLE]\n{json.dumps(few_shot, indent=2)}\n"
            f"[END_EXAMPLE]\n\n"
            f"Generate a new crisis for a {domain_hint} scenario. "
            f"Difficulty: {difficulty}. Question type: {question_type}."
        )

        for attempt in range(1, self.max_retries + 1):
            try:
                result = self._call_gpt4o(
                    system_prompt=SYSTEM_PROMPT_B1,
                    user_prompt=user_prompt,
                    json_mode=True)

                if not result['success']:
                    logger.warning(
                        f"B1 attempt {attempt}/{self.max_retries} "
                        f"failed: {result['error']}")
                    self.cost_tracker.retries += 1
                    time.sleep(self.retry_delay_base ** attempt)
                    continue

                # Parse mutation spec
                parsed = result['parsed']
                mutations = parsed.get('mutations', [])
                sketch = parsed.get('narrative_sketch', '')

                # Validate mutation spec structure
                validation = self._validate_mutation_spec_structure(
                    mutations, ctype, allowed_ops)
                if not validation['valid']:
                    logger.warning(
                        f"B1 attempt {attempt}: invalid spec: "
                        f"{validation['reason']}")
                    self.cost_tracker.retries += 1
                    time.sleep(self.retry_delay_base ** attempt)
                    continue

                return ContextualizerResult(
                    success=True,
                    mutation_spec=mutations,
                    narrative_context=sketch,
                    attempts=attempt,
                    total_tokens=result['total_tokens'],
                    cost_usd=result['cost'])

            except Exception as e:
                logger.error(f"B1 attempt {attempt} exception: {e}")
                self.cost_tracker.retries += 1
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay_base ** attempt)

        return ContextualizerResult(
            success=False,
            error=f"Failed after {self.max_retries} attempts",
            attempts=self.max_retries)

    # ── Step B2: CONTEXT + DISRUPTION narrative ───────────────────────────

    def generate_narrative(
            self, resources_tasks_text: str,
            resolved_mutations: list[dict],
            narrative_sketch: str,
            ctype: str,
            domain_hint: Optional[str] = None
    ) -> ContextualizerResult:
        """
        Step B2: pre-written RESOURCES+TASKS + resolved mutations
                 → CONTEXT + DISRUPTION text.
        GPT-4o sees exact numerical values from resolved mutations.
        It writes narrative AROUND these facts — never invents numbers.
        """
        import random

        if domain_hint is None:
            domains = DOMAIN_DESCRIPTIONS.get(ctype,
                                              DOMAIN_DESCRIPTIONS["TYPE-A"])
            domain_hint = random.choice(domains)

        # Format resolved mutations as readable text
        mutation_descriptions = []
        for m in resolved_mutations:
            mutation_descriptions.append(
                self._format_resolved_mutation(m))

        user_prompt = (
            f"[DOMAIN: {domain_hint}]\n"
            f"[NARRATIVE SKETCH: {narrative_sketch}]\n\n"
            f"[RESOLVED_MUTATIONS]\n"
            + "\n".join(f"- {d}" for d in mutation_descriptions)
            + "\n[END MUTATIONS]\n\n"
            f"[PRE-WRITTEN STRUCTURE]\n{resources_tasks_text}\n"
            f"[END STRUCTURE]\n\n"
            f"Write CONTEXT and DISRUPTION sections only. "
            f"Use the exact numbers from [RESOLVED_MUTATIONS]."
        )

        for attempt in range(1, self.max_retries + 1):
            try:
                result = self._call_gpt4o(
                    system_prompt=SYSTEM_PROMPT_B2,
                    user_prompt=user_prompt,
                    json_mode=False)  # plain text, not JSON

                if not result['success']:
                    logger.warning(
                        f"B2 attempt {attempt}/{self.max_retries} "
                        f"failed: {result['error']}")
                    self.cost_tracker.retries += 1
                    time.sleep(self.retry_delay_base ** attempt)
                    continue

                text = result['text']

                # Validate: must contain CONTEXT: and DISRUPTION:
                if 'context:' not in text.lower():
                    logger.warning(f"B2 attempt {attempt}: "
                                   f"missing CONTEXT section")
                    self.cost_tracker.retries += 1
                    continue
                if 'disruption:' not in text.lower():
                    logger.warning(f"B2 attempt {attempt}: "
                                   f"missing DISRUPTION section")
                    self.cost_tracker.retries += 1
                    continue

                # Extract sections
                context, disruption = self._extract_sections(text)

                return ContextualizerResult(
                    success=True,
                    narrative_context=context,
                    narrative_disruption=disruption,
                    attempts=attempt,
                    total_tokens=result['total_tokens'],
                    cost_usd=result['cost'])

            except Exception as e:
                logger.error(f"B2 attempt {attempt} exception: {e}")
                self.cost_tracker.retries += 1
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay_base ** attempt)

        return ContextualizerResult(
            success=False,
            error=f"Narrative generation failed after "
                  f"{self.max_retries} attempts",
            attempts=self.max_retries)

    # ── GPT-4o API call (shared by B1 and B2) ────────────────────────────

    def _call_gpt4o(self, system_prompt: str, user_prompt: str,
                    json_mode: bool = False) -> dict:
        """
        Single GPT-4o API call with cost tracking.
        Returns: {success, parsed/text, total_tokens, cost, error}
        """
        client = self._get_client()

        kwargs = {
            'model': self.model,
            'messages': [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            'temperature': self.temperature,
            'max_tokens': 1000,
        }
        if json_mode:
            kwargs['response_format'] = {"type": "json_object"}

        try:
            response = client.chat.completions.create(**kwargs)
            choice = response.choices[0]
            text = choice.message.content.strip()

            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            total_tokens = input_tokens + output_tokens

            cost = (
                input_tokens / 1000 * self.cost_tracker.INPUT_COST_PER_1K +
                output_tokens / 1000 * self.cost_tracker.OUTPUT_COST_PER_1K
            )
            self.cost_tracker.record(input_tokens, output_tokens, True)

            if json_mode:
                # Parse JSON — strip markdown fences if present
                clean = re.sub(r'^```json\s*', '', text)
                clean = re.sub(r'\s*```$', '', clean).strip()
                try:
                    parsed = json.loads(clean)
                    return {'success': True, 'parsed': parsed,
                            'text': text,
                            'total_tokens': total_tokens,
                            'cost': cost}
                except json.JSONDecodeError as e:
                    self.cost_tracker.record(0, 0, False)
                    return {'success': False,
                            'error': f'JSON parse error: {e}. '
                                     f'Raw: {text[:200]}',
                            'total_tokens': total_tokens,
                            'cost': cost}
            else:
                return {'success': True, 'text': text,
                        'total_tokens': total_tokens,
                        'cost': cost}

        except Exception as e:
            self.cost_tracker.record(0, 0, False)
            return {'success': False,
                    'error': f'API error: {type(e).__name__}: {str(e)[:200]}',
                    'total_tokens': 0, 'cost': 0.0}

    # ── Validation helpers ────────────────────────────────────────────────

    @staticmethod
    def _validate_mutation_spec_structure(
            mutations: list, ctype: str,
            allowed_ops: list) -> dict:
        """
        Validates the structure of GPT-4o's mutation spec output.
        Does NOT validate values (that is the validator's job after
        severity resolution). Only checks structural correctness.
        """
        if not isinstance(mutations, list):
            return {'valid': False, 'reason': 'mutations is not a list'}
        if len(mutations) == 0:
            return {'valid': False, 'reason': 'mutations list is empty'}

        for i, m in enumerate(mutations):
            if not isinstance(m, dict):
                return {'valid': False,
                        'reason': f'mutation [{i}] is not a dict'}
            if 'op' not in m:
                return {'valid': False,
                        'reason': f'mutation [{i}] missing "op" field'}
            if m['op'] not in allowed_ops:
                return {'valid': False,
                        'reason': f'mutation [{i}]: op "{m["op"]}" '
                                  f'not in allowed ops for {ctype}'}

            # Op-specific structural checks
            op = m['op']
            if op in ('reduce_capacity', 'freeze_resource'):
                if 'resource' not in m:
                    return {'valid': False,
                            'reason': f'mutation [{i}]: {op} '
                                      f'missing "resource" field'}
            elif op == 'enforce_time_lag':
                if 'i' not in m or 'j' not in m:
                    return {'valid': False,
                            'reason': f'mutation [{i}]: enforce_time_lag '
                                      f'missing "i" or "j" field'}
            elif op == 'restrict_modes':
                if 'activity_id' not in m:
                    return {'valid': False,
                            'reason': f'mutation [{i}]: restrict_modes '
                                      f'missing "activity_id" field'}
                if 'remove_mode_ids' not in m:
                    return {'valid': False,
                            'reason': f'mutation [{i}]: restrict_modes '
                                      f'missing "remove_mode_ids" field'}
            elif op == 'reduce_nonrenewable_budget':
                if 'resource' not in m:
                    return {'valid': False,
                            'reason': f'mutation [{i}]: '
                                      f'reduce_nonrenewable_budget '
                                      f'missing "resource" field'}

        return {'valid': True, 'reason': None}

    @staticmethod
    def _format_resolved_mutation(m: dict) -> str:
        """
        Formats a resolved mutation dict as a human-readable string
        for the B2 prompt. Uses exact numerical values.
        """
        op = m.get('op', 'unknown')
        if op == 'reduce_capacity':
            return (f"Resource {m['resource']} capacity reduced by "
                    f"{m['delta']} units from t={m['t_start']} to "
                    f"t={m['t_end']}")
        elif op == 'freeze_resource':
            return (f"Resource {m['resource']} fully offline from "
                    f"t={m['t_start']} to t={m['t_end']}")
        elif op == 'enforce_time_lag':
            s = (f"Minimum {m['relation']} lag of {m['lag_min']} "
                 f"time units between T{m['i']} and T{m['j']}")
            if m.get('lag_max') is not None:
                s += (f" (maximum lag: {m['lag_max']} time units — "
                      f"T{m['j']} must start within this window)")
            return s
        elif op == 'add_precedence':
            return (f"New {m.get('relation', 'FS')} precedence: "
                    f"T{m['i']} must complete before T{m['j']} starts")
        elif op == 'alter_mode_profile':
            return (f"Activity T{m['activity_id']} mode {m['mode_id']} "
                    f"duration changed by {m['duration_delta']} units")
        elif op == 'restrict_modes':
            return (f"Activity T{m['activity_id']} restricted: "
                    f"modes {m['remove_mode_ids']} no longer available")
        elif op == 'add_mode':
            return (f"Activity T{m['activity_id']} gains new mode "
                    f"(duration={m['duration']})")
        elif op == 'reduce_nonrenewable_budget':
            return (f"Nonrenewable resource {m['resource']} budget "
                    f"reduced by {m['delta']} units")
        elif op == 'alter_objective':
            return f"Objective changed to: {m.get('obj_type', 'unknown')}"
        else:
            return f"Mutation: {json.dumps(m)}"

    @staticmethod
    def _extract_sections(text: str) -> tuple[str, str]:
        """
        Extracts CONTEXT and DISRUPTION sections from GPT-4o output.
        Returns (context_text, disruption_text).
        """
        context = ""
        disruption = ""

        # Try to split on section headers
        parts = re.split(r'(?i)(CONTEXT:|DISRUPTION:)', text)

        # parts should be: ['', 'CONTEXT:', '...', 'DISRUPTION:', '...']
        for i, part in enumerate(parts):
            if re.match(r'(?i)CONTEXT:', part):
                if i + 1 < len(parts):
                    context = parts[i + 1].strip()
            elif re.match(r'(?i)DISRUPTION:', part):
                if i + 1 < len(parts):
                    disruption = parts[i + 1].strip()

        return context, disruption
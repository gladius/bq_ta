"""Static vs dynamic segmentation, at two granularities that answer different questions.

  region - which whole lines are regenerated per call. Answers "where does the cached prefix
           break". Document frequency does this at F1 0.994.
  value  - which characters inside such a line actually differ. Answers "how much is really
           payload", and gives a human-readable template. Drain does this at F1 0.987.

They are not alternatives: measured on the same labelled spans, document frequency scores 0.420
at value level and Drain scores 0.420 at region level. Run both, for different outputs.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

STATIC_FRACTION = 0.9
WILDCARD = "<*>"
DRAIN_SIM_TH = 0.4


def lines_of(text: str) -> List[str]:
    return [ln.strip() for ln in (text or "").split("\n") if ln.strip()]


def static_dynamic(prompts: Sequence[str], fraction: float = STATIC_FRACTION
                   ) -> Tuple[set, set, Counter]:
    df: Counter = Counter()
    for prompt in prompts:
        df.update(set(lines_of(prompt)))
    threshold = fraction * len(prompts)
    static = {ln for ln, count in df.items() if count >= threshold}
    return static, {ln for ln in df if ln not in static}, df


def common_prefix(strings: Sequence[str]) -> str:
    """Longest character prefix shared by every string - what a provider cache can key on."""
    if not strings:
        return ""
    low, high = min(strings), max(strings)
    for i, ch in enumerate(low):
        if i >= len(high) or high[i] != ch:
            return low[:i]
    return low


# ---------------------------------------------------------------------------------------
# value-level templates
# ---------------------------------------------------------------------------------------

def _miner(sim_th: float):
    from drain3 import TemplateMiner
    from drain3.template_miner_config import TemplateMinerConfig

    config = TemplateMinerConfig()
    config.drain_sim_th = sim_th
    config.drain_depth = 4
    return TemplateMiner(config=config)


def wildcard_spans(template: str, line: str) -> List[Tuple[int, int]]:
    """Align a Drain template to a line and return the character ranges of its wildcards."""
    if not template or WILDCARD not in template:
        return []
    literals = template.split(WILDCARD)
    spans: List[Tuple[int, int]] = []
    cursor = 0
    head = literals[0]
    if head:
        if not line.startswith(head):
            return []
        cursor = len(head)
    for literal in literals[1:]:
        if literal == "":
            if cursor < len(line):
                spans.append((cursor, len(line)))
            break
        index = line.find(literal, cursor)
        if index < 0:
            return []
        if index > cursor:
            spans.append((cursor, index))
        cursor = index + len(literal)
    return spans


@dataclass
class DynamicField:
    """One varying slot inside an otherwise stable line."""

    template: str                 # e.g. "User: <*> | Date: <*>"
    occurrences: int
    examples: List[str] = field(default_factory=list)
    avg_value_chars: int = 0
    scaffold_chars: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return {"template": self.template, "occurrences": self.occurrences,
                "examples": self.examples[:3], "avg_value_chars": self.avg_value_chars,
                "scaffold_chars": self.scaffold_chars}


def induce_fields(dynamic_lines: Sequence[str], sim_th: float = DRAIN_SIM_TH,
                  max_lines: int = 4000) -> List[DynamicField]:
    """Turn the dynamic lines of a node into readable templates with <*> slots."""
    corpus = [ln for ln in dynamic_lines if ln][:max_lines]
    if not corpus:
        return []
    try:
        miner = _miner(sim_th)
    except Exception:                       # drain3 optional; segmentation still works
        return []
    for line in corpus:
        miner.add_log_message(line)

    grouped: Dict[str, List[Tuple[str, List[Tuple[int, int]]]]] = {}
    for line in corpus:
        cluster = miner.match(line)
        if cluster is None:
            continue
        template = cluster.get_template()
        spans = wildcard_spans(template, line)
        grouped.setdefault(template, []).append((line, spans))

    fields: List[DynamicField] = []
    for template, entries in sorted(grouped.items(), key=lambda kv: -len(kv[1])):
        values = [line[a:b] for line, spans in entries for a, b in spans]
        scaffold = sum(len(part) for part in template.split(WILDCARD))
        fields.append(DynamicField(
            template=template,
            occurrences=len(entries),
            examples=[v for v in values[:3] if v],
            avg_value_chars=int(sum(len(v) for v in values) / len(values)) if values else 0,
            scaffold_chars=scaffold,
        ))
    return fields


@dataclass
class Segmentation:
    static_lines: List[str]
    dynamic_lines: List[str]
    fields: List[DynamicField]
    static_chars: int
    dynamic_chars: int

    def as_dict(self) -> Dict[str, Any]:
        return {
            "static_line_count": len(self.static_lines),
            "dynamic_line_count": len(self.dynamic_lines),
            "static_chars": self.static_chars,
            "dynamic_chars": self.dynamic_chars,
            "fields": [f.as_dict() for f in self.fields[:12]],
        }


def segment(prompts: Sequence[str], fraction: float = STATIC_FRACTION) -> Segmentation:
    static, dynamic, df = static_dynamic(prompts, fraction)
    modal = Counter(prompts).most_common(1)[0][0] if prompts else ""
    ordered_static = [ln for ln in lines_of(modal) if ln in static]
    dyn_instances = [ln for prompt in prompts for ln in lines_of(prompt) if ln in dynamic]
    return Segmentation(
        static_lines=ordered_static,
        dynamic_lines=sorted(dynamic, key=lambda ln: -df[ln])[:200],
        fields=induce_fields(dyn_instances),
        static_chars=sum(len(ln) for ln in static),
        dynamic_chars=sum(len(ln) for ln in dynamic),
    )

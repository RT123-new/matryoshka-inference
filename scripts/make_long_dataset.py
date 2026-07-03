from __future__ import annotations

import argparse
import json
from pathlib import Path


DISTRACTORS = [
    "Facilities note: the lobby plants were watered, visitor badges were reordered, and the third-floor whiteboard markers were replaced.",
    "Operations note: weekly lunch moved to noon, the office map was refreshed, and the old printer queue was renamed.",
    "Archive note: historical meeting minutes mention office furniture, onboarding snacks, and a retired calendar link.",
    "Procurement note: the stationery supplier shipped envelopes, keyboard covers, cable ties, and spare monitor arms.",
    "People note: a voluntary survey asked about lighting, desk height, commute preferences, and meeting room temperature.",
    "Release note: unrelated UI copy changed in a legacy screen that is not part of the question being asked.",
    "Support note: a test ticket referenced a sample customer, a mock invoice, and placeholder screenshots for training.",
    "Security note: an old awareness poster reminded staff to lock screens and report suspicious emails.",
    "Finance note: a sample spreadsheet used fictional values for training and must not be used as source evidence.",
    "Research note: a draft memo discussed benchmark methodology but did not mention the answer-critical facts.",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a long-context variant of a JSONL benchmark dataset.")
    parser.add_argument("--source", default="data/tasks/synthetic.jsonl")
    parser.add_argument("--out", default="data/tasks/synthetic_long.jsonl")
    parser.add_argument("--repeat", type=int, default=12, help="Number of distractor paragraphs to add before and after.")
    args = parser.parse_args()

    source = Path(args.source)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for line in source.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        before = []
        after = []
        for idx in range(args.repeat):
            before.append(f"Distractor before {idx + 1}. {DISTRACTORS[idx % len(DISTRACTORS)]}")
            after.append(f"Distractor after {idx + 1}. {DISTRACTORS[(idx + 3) % len(DISTRACTORS)]}")
        record["id"] = f"{record['id']}_long"
        record["document"] = "\n\n".join(
            [
                "LONG CONTEXT SYNTHETIC DOCUMENT",
                *before,
                "ANSWER-RELEVANT SOURCE SECTION",
                record["document"],
                "END ANSWER-RELEVANT SOURCE SECTION",
                *after,
            ]
        )
        metadata = dict(record.get("metadata", {}))
        metadata.update(
            {
                "long_context_variant": True,
                "distractor_paragraphs_before": args.repeat,
                "distractor_paragraphs_after": args.repeat,
                "source_dataset": str(source),
            }
        )
        record["metadata"] = metadata
        records.append(record)

    with out.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    print(f"Wrote {len(records)} long-context tasks to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from sclab.utils.jsonl import read_jsonl, write_jsonl


@dataclass
class BenchmarkTask:
    id: str
    type: str
    document: str
    question: str
    gold_answer: str
    answer_aliases: list[str] = field(default_factory=list)
    must_include: list[str] = field(default_factory=list)
    must_not_include: list[str] = field(default_factory=list)
    source_span: str = ""
    difficulty: str = "medium"
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> BenchmarkTask:
        required = ["id", "type", "document", "question", "gold_answer"]
        missing = [key for key in required if key not in record]
        if missing:
            raise ValueError(f"Task record missing required fields: {', '.join(missing)}")
        return cls(
            id=str(record["id"]),
            type=str(record["type"]),
            document=str(record["document"]),
            question=str(record["question"]),
            gold_answer=str(record["gold_answer"]),
            answer_aliases=[str(item) for item in record.get("answer_aliases", [])],
            must_include=[str(item) for item in record.get("must_include", [])],
            must_not_include=[str(item) for item in record.get("must_not_include", [])],
            source_span=str(record.get("source_span", "")),
            difficulty=str(record.get("difficulty", "medium")),
            metadata=dict(record.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_tasks(path: str | Path, max_tasks: int | None = None) -> list[BenchmarkTask]:
    tasks = [BenchmarkTask.from_record(record) for record in read_jsonl(path)]
    if max_tasks is not None:
        return tasks[:max_tasks]
    return tasks


def write_tasks(path: str | Path, tasks: list[BenchmarkTask]) -> None:
    write_jsonl(path, [task.to_dict() for task in tasks])


def ingest_documents(input_path: str | Path, out_path: str | Path) -> list[BenchmarkTask]:
    root = Path(input_path)
    if root.is_file():
        paths = [root]
    else:
        paths = sorted(
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in {".txt", ".md", ".json", ".py", ".yaml", ".yml"}
        )
    tasks: list[BenchmarkTask] = []
    for path in paths:
        text = path.read_text(encoding="utf-8", errors="replace")
        digest = hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:8]
        tasks.append(
            BenchmarkTask(
                id=f"user_{path.stem}_{digest}",
                type="manual_user_document",
                document=text,
                question="Manual QA prompt: ask a source-grounded question about this document.",
                gold_answer="",
                source_span="",
                difficulty="unknown",
                metadata={
                    "source_path": str(path),
                    "manual_review_required": True,
                    "note": ("No local model-generated task was created; "
                             "benchmark compression ratios or use sclab single."),
                },
            )
        )
    write_tasks(out_path, tasks)
    return tasks

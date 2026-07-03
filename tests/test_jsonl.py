from sclab.utils.jsonl import read_jsonl, write_jsonl


def test_jsonl_roundtrip(tmp_path):
    path = tmp_path / "records.jsonl"
    records = [{"id": "a", "value": 1}, {"id": "b", "value": 2}]
    write_jsonl(path, records)
    assert list(read_jsonl(path)) == records

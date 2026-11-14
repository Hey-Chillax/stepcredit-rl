"""Dump the history of an offline W&B run (`outputs/<run>/wandb/offline-run-*/run-*.wandb`) to CSV, one row per step.

    python -m tools.wandb_history <run-xxxx.wandb> <out.csv>

Needs the `wandb` package of the training environment. It reads the transaction log with wandb's own datastore reader,
so it works without a W&B account or network; a half-written last block (run still writing) ends the scan early.
"""

from __future__ import annotations

import csv
import json
import sys


def main(argv: list[str] | None = None) -> None:
    from wandb.proto import wandb_internal_pb2 as pb
    from wandb.sdk.internal import datastore

    src, dst = (argv or sys.argv[1:])[:2]
    ds = datastore.DataStore()
    ds.open_for_scan(src)
    rows: dict[int, dict[str, object]] = {}
    while True:
        try:
            data = ds.scan_data()
        except Exception as exc:  # truncated tail of a run that is still being written
            print(f"stopped at a partial record: {type(exc).__name__}", file=sys.stderr)
            break
        if data is None:
            break
        record = pb.Record()
        record.ParseFromString(data)
        if record.WhichOneof("record_type") != "history":
            continue
        row: dict[str, object] = {}
        for item in record.history.item:
            key = item.key or "/".join(item.nested_key)
            try:
                row[key] = json.loads(item.value_json)
            except json.JSONDecodeError:
                continue
        step = row.get("_step", record.history.step.num)
        rows.setdefault(int(step), {}).update(row)

    keys = sorted(
        {k for r in rows.values() for k, v in r.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
    )
    with open(dst, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["step", *keys])
        for step in sorted(rows):
            writer.writerow([step, *(rows[step].get(k, "") for k in keys)])
    print(f"{len(rows)} steps ({min(rows)}..{max(rows)}), {len(keys)} numeric keys -> {dst}")


if __name__ == "__main__":
    main()

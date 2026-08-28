"""Freeze the freefall/ramp engineering-pilot trajectory snapshot.

Reads the immutable schedule and object split, enumerates every scheduled
pilot trajectory, applies the eligibility gates, and writes a frozen trajectory
manifest plus a human-readable audit report. Repeating against unchanged inputs
produces byte-identical manifest contents and summary counts.
"""

from __future__ import annotations

import argparse
import os

from data.trajectory_manifest import (
    DEFAULT_MIN_FRAMES,
    ManifestConfig,
    build_manifest,
    render_audit,
    write_json,
)

DEFAULT_DATA_ROOT = os.environ.get("TORQUE_DATA_ROOT", "/data2/adrien/TORQUE")
DEFAULT_SCENARIOS = ("freefall", "ramp")
DEFAULT_OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "docs",
    "manifests",
    "pilot_v1",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--schedule",
        default=os.path.join(DEFAULT_DATA_ROOT, "v2", "manifests", "ordinary_schedule.json"),
    )
    parser.add_argument(
        "--split",
        default=os.path.join(DEFAULT_DATA_ROOT, "v2", "manifests", "object_split.json"),
    )
    parser.add_argument("--scenarios", nargs="+", default=list(DEFAULT_SCENARIOS))
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dataset-version", default=None)
    parser.add_argument("--min-frames", type=int, default=None)
    parser.add_argument("--generated-at", default=None, help="optional ISO-8601 stamp for provenance (omitted by default to keep the manifest deterministic)")
    args = parser.parse_args(argv)

    config = ManifestConfig(
        data_root=args.data_root,
        schedule_path=args.schedule,
        split_path=args.split,
        scenarios=tuple(args.scenarios),
        geometry_dir=os.path.join(args.data_root, "assets"),
        utonia_dir=os.path.join(args.data_root, "3D_features"),
        dataset_version=args.dataset_version,
        min_frames=args.min_frames if args.min_frames is not None else DEFAULT_MIN_FRAMES,
        scope="pilot-freefall-ramp",
        generated_at=args.generated_at,
    )

    manifest = build_manifest(config)
    manifest_path = os.path.join(args.output_dir, "trajectory_manifest.json")
    audit_path = os.path.join(args.output_dir, "audit.md")

    write_json(manifest, manifest_path)
    with open(audit_path, "w") as f:
        f.write(render_audit(manifest))

    summary = manifest["summary"]
    print(
        f"froze {summary['included']} trajectories "
        f"({summary['excluded']} excluded of {summary['total_scheduled']} scheduled)"
    )
    print(f"manifest: {manifest_path}")
    print(f"audit:    {audit_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

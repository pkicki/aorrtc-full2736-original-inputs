#!/usr/bin/env python3
"""Verify a hydrated public AORRTC release against its complete manifest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

COUNTS = {"mbm_panda": 650, "mpinets_panda": 1519, "mbm_ur5": 567}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(root: Path) -> dict:
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("schema") != "aorrtc_full2736_public_research_release_v2":
        raise ValueError("unexpected release manifest schema")
    verified = 0
    for relative, record in manifest["files"].items():
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(f"missing or unsafe release file: {relative}")
        with path.open("rb") as stream:
            if stream.read(80).startswith(
                b"version https://git-lfs.github.com/spec/v1"
            ):
                raise ValueError(f"unhydrated Git LFS pointer: {relative}")
        if (
            path.stat().st_size != record["size_bytes"]
            or digest(path) != record["sha256"]
        ):
            raise ValueError(f"size or SHA256 mismatch: {relative}")
        verified += 1
    for track, expected in COUNTS.items():
        ids_path = root / "inputs" / track / "exported_ids.txt"
        ids = [
            line.strip() for line in ids_path.read_text().splitlines() if line.strip()
        ]
        if len(ids) != expected or len(set(ids)) != expected:
            raise ValueError(f"source-order IDs invalid for {track}: {len(ids)}")
    identity = json.loads(
        (root / "provenance/source_identity.public_projection.json").read_text()
    )
    contracts = json.loads(
        (root / "limits/full_c3_limits.public_projection.json").read_text()
    )
    if identity["record_status"] != "REGENERATED_PUBLIC_PROJECTION_NOT_HISTORICAL_FILE":
        raise ValueError("source identity projection is not clearly labeled")
    if (
        contracts["record_status"]
        != "REGENERATED_PUBLIC_PROJECTION_NOT_HISTORICAL_FILE"
    ):
        raise ValueError("full-C3 projection is not clearly labeled")
    return {
        "status": "PUBLIC_RELEASE_FILES_VERIFIED",
        "files": verified,
        "path_count": sum(COUNTS.values()),
        "manifest_sha256": digest(root / "manifest.json"),
    }


if __name__ == "__main__":
    base = Path(__file__).resolve().parent
    root = base if (base / "manifest.json").is_file() else base.parent
    print(json.dumps(verify(root), sort_keys=True, indent=2))

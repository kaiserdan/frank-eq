#!/usr/bin/env python3
"""Materialize the initial Frank-EQ tree from temporary payload chunks."""
from __future__ import annotations

import base64
import io
import os
from pathlib import Path, PurePosixPath
import tarfile


def main() -> None:
    root = Path.cwd().resolve()
    parts = sorted((root / ".bootstrap_payload").glob("*.part"))
    if not parts:
        raise RuntimeError("bootstrap payload is missing")
    encoded = "".join(path.read_text().strip() for path in parts)
    raw = base64.b64decode(encoded.encode("ascii"), validate=True)
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as archive:
        members = archive.getmembers()
        for member in members:
            rel = PurePosixPath(member.name)
            if rel.is_absolute() or ".." in rel.parts or not member.isfile():
                raise RuntimeError(f"unsafe archive member: {member.name}")
            target = root.joinpath(*rel.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source is None:
                raise RuntimeError(f"missing payload: {member.name}")
            target.write_bytes(source.read())
            os.chmod(target, member.mode)
    print(f"materialized {len(members)} files")


if __name__ == "__main__":
    main()

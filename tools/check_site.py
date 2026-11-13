#!/usr/bin/env python3
"""Read-only checks for the buildless cigAMT website (standard library only).

Run from any working directory:
    python /path/to/cigAMT/tools/check_site.py

This script does not start a server, change files, inspect original research
directories, or make network requests. The expected result inventory is pinned
to the current manuscript: 8 tasks, 66 model rows, and 238 metric cells.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import struct
import sys
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit


REPO = Path(__file__).resolve().parents[1]
DOCS = REPO / "docs"
EXPECTED_TASKS = {
    "sferic", "classification", "localization", "denoising",
    "frequency", "forecasting", "interpolation", "imputation",
}
TEXT_SUFFIXES = {
    ".html", ".css", ".js", ".json", ".svg", ".txt", ".md",
    ".xml", ".csv", ".webmanifest",
}
FORBIDDEN_SUFFIXES = {
    ".npy", ".npz", ".h5", ".hdf5", ".mat", ".pt", ".pth",
    ".ckpt", ".safetensors", ".zip", ".7z", ".tar", ".gz",
}
FORBIDDEN_DIRS = {
    ".git", "__pycache__", "raw-data", "raw_data", "checkpoints", "ckpts",
    "runs", "private", "private-release", "release-prep", "release_staging",
}
PRIVATE_PATH = re.compile(
    r"(?<![A-Za-z0-9])(?:[A-Za-z]:[\\/]|file://)"
    r"|(?<![:\w])/(?:Users|home)/[A-Za-z0-9_.-]+/",
    re.IGNORECASE,
)


class Document(HTMLParser):
    def __init__(self, path: Path) -> None:
        super().__init__(convert_charrefs=True)
        self.path = path
        self.ids: list[str] = []
        self.links: list[tuple[str, int, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        for key, value in attrs:
            if value is None:
                continue
            if key == "id":
                self.ids.append(value)
            if key in {"href", "src", "data-image"}:
                self.links.append((value, self.getpos()[0], key))
            elif key == "srcset":
                # The site does not use data-URL srcsets. Each candidate's first
                # whitespace-delimited item is a local or external image URL.
                for item in value.split(","):
                    parts = item.strip().split()
                    if parts:
                        self.links.append((parts[0], self.getpos()[0], key))

    handle_startendtag = handle_starttag


class Checks:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.stats: dict[str, int] = {}
        self.text: dict[Path, str] = {}
        self.html: dict[Path, Document] = {}

    def error(self, message: str) -> None:
        self.errors.append(message)

    def relative(self, path: Path) -> str:
        try:
            return path.relative_to(REPO).as_posix()
        except ValueError:
            return str(path)

    @staticmethod
    def within(path: Path, directory: Path) -> bool:
        try:
            path.relative_to(directory)
            return True
        except ValueError:
            return False

    def read_json(self, path: Path):
        def reject_constant(value: str):
            raise ValueError(f"non-finite JSON constant {value}")

        try:
            text = self.text.get(path)
            if text is None:
                text = path.read_text(encoding="utf-8-sig")
            return json.loads(text, parse_constant=reject_constant)
        except (OSError, UnicodeError, ValueError) as exc:
            self.error(f"{self.relative(path)}: invalid JSON/UTF-8 ({exc})")
            return None

    def files_and_encoding(self) -> None:
        if not DOCS.is_dir():
            self.error("docs/ does not exist")
            return
        files = sorted(path for path in DOCS.rglob("*") if path.is_file())
        self.stats["public files"] = len(files)
        for path in files:
            rel = path.relative_to(DOCS)
            if not self.within(path.resolve(), DOCS.resolve()):
                self.error(f"{self.relative(path)}: resolves outside docs/")
                continue
            if path.suffix.lower() in FORBIDDEN_SUFFIXES:
                self.error(f"{self.relative(path)}: scientific array, model, or archive in public docs/")
            if any(part.lower() in FORBIDDEN_DIRS for part in rel.parts[:-1]):
                self.error(f"{self.relative(path)}: private/research directory in public docs/")
            if "local_provenance" in path.name.lower() or path.name.startswith(".env"):
                self.error(f"{self.relative(path)}: private provenance or environment file")
            if path.suffix.lower() in TEXT_SUFFIXES or path.name == ".nojekyll":
                try:
                    text = path.read_text(encoding="utf-8-sig")
                except (OSError, UnicodeError) as exc:
                    self.error(f"{self.relative(path)}: cannot read as UTF-8 ({exc})")
                    continue
                self.text[path] = text
                if "\ufffd" in text:
                    self.error(f"{self.relative(path)}: Unicode replacement character found")
                if PRIVATE_PATH.search(text):
                    self.error(f"{self.relative(path)}: local filesystem path found in public text")
                if "local_provenance" in text.lower():
                    self.error(f"{self.relative(path)}: private provenance reference in public text")
        self.stats["UTF-8 text files"] = len(self.text)

    def html_documents(self) -> None:
        for path, text in self.text.items():
            if path.suffix.lower() != ".html":
                continue
            doc = Document(path)
            try:
                doc.feed(text)
                doc.close()
            except Exception as exc:
                self.error(f"{self.relative(path)}: HTML parser error ({exc})")
                continue
            self.html[path.resolve()] = doc
            for identifier, count in Counter(doc.ids).items():
                if count > 1:
                    self.error(f"{self.relative(path)}: duplicate id={identifier!r} ({count} times)")
                if not identifier:
                    self.error(f"{self.relative(path)}: empty HTML id")
        self.stats["HTML pages"] = len(self.html)
        self.stats["unique HTML IDs"] = sum(len(set(doc.ids)) for doc in self.html.values())

    def check_link(self, source: Path, value: str, line: int, attribute: str) -> None:
        value = value.strip()
        label = f"{self.relative(source)}:{line} {attribute}={value!r}"
        if not value:
            self.error(f"{label}: empty URL")
            return
        try:
            url = urlsplit(value)
        except ValueError as exc:
            self.error(f"{label}: invalid URL ({exc})")
            return
        if url.scheme or url.netloc:
            if url.scheme.lower() not in {"", "https", "http", "mailto", "tel", "data"}:
                self.error(f"{label}: unsupported URL scheme")
            return
        self.stats["local link/asset references"] += 1
        decoded = unquote(url.path)
        # A leading slash is resolved against the published docs root. This
        # validates existence; relative URLs remain preferable on project Pages.
        target = ((DOCS / decoded.lstrip("/")) if decoded.startswith("/")
                  else source.parent / decoded) if decoded else source
        target = target.resolve()
        if not self.within(target, DOCS.resolve()):
            self.error(f"{label}: target escapes docs/")
            return
        if target.is_dir():
            target = target / "index.html"
        if not target.is_file():
            self.error(f"{label}: missing local file {self.relative(target)}")
            return
        if url.fragment and target.suffix.lower() == ".html":
            self.stats["local HTML fragment references"] += 1
            doc = self.html.get(target.resolve())
            fragment = unquote(url.fragment)
            if doc is None or fragment not in doc.ids:
                self.error(f"{label}: missing target id={fragment!r}")

    def local_links(self) -> None:
        self.stats["local link/asset references"] = 0
        self.stats["local HTML fragment references"] = 0
        for doc in self.html.values():
            for value, line, attr in doc.links:
                self.check_link(doc.path, value, line, attr)
        # CSS URL references are relative to the stylesheet's own directory.
        for path, text in self.text.items():
            if path.suffix.lower() == ".css":
                for match in re.finditer(r"url\(\s*['\"]?([^)'\"]+)['\"]?\s*\)", text):
                    self.check_link(path, match.group(1), text.count("\n", 0, match.start()) + 1, "url")

    @staticmethod
    def finite_number(value) -> bool:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return False
        try:
            return math.isfinite(value)
        except (OverflowError, TypeError):
            return False

    def benchmark_json(self) -> None:
        for path in self.text:
            if path.suffix.lower() == ".json":
                self.read_json(path)
        data = self.read_json(DOCS / "data" / "benchmarks.json")
        if not isinstance(data, dict) or not isinstance(data.get("tasks"), list):
            self.error("docs/data/benchmarks.json: expected an object containing tasks[]")
            return
        tasks = data["tasks"]
        ids = [task.get("id") for task in tasks if isinstance(task, dict)]
        if len(tasks) != 8 or len(ids) != 8 or set(ids) != EXPECTED_TASKS:
            self.error("benchmark inventory must contain the eight unique expected task IDs")
        row_count = cell_count = 0
        for index, task in enumerate(tasks):
            if not isinstance(task, dict):
                self.error(f"task {index}: expected an object")
                continue
            task_id = task.get("id", index)
            metrics, rows = task.get("metrics"), task.get("rows")
            if not isinstance(metrics, list) or not isinstance(rows, list):
                self.error(f"task {task_id}: expected metrics[] and rows[]")
                continue
            keys = []
            for metric in metrics:
                if not isinstance(metric, dict) or not isinstance(metric.get("key"), str):
                    self.error(f"task {task_id}: invalid metric definition")
                    continue
                keys.append(metric["key"])
                if metric.get("direction") not in {"min", "max"}:
                    self.error(f"task {task_id}/{metric['key']}: invalid preferred direction")
                decimals = metric.get("decimals")
                if isinstance(decimals, bool) or not isinstance(decimals, int) or not 0 <= decimals <= 10:
                    self.error(f"task {task_id}/{metric['key']}: invalid decimal precision")
            if len(set(keys)) != len(keys):
                self.error(f"task {task_id}: duplicate metric keys")
            models = []
            row_count += len(rows)
            for row_index, row in enumerate(rows):
                if not isinstance(row, dict) or not isinstance(row.get("values"), dict):
                    self.error(f"task {task_id}/row {row_index}: invalid row")
                    continue
                model = row.get("model")
                if not isinstance(model, str) or not model.strip():
                    self.error(f"task {task_id}/row {row_index}: missing model name")
                else:
                    models.append(model)
                values = row["values"]
                if set(values) != set(keys):
                    self.error(f"task {task_id}/{model}: row keys differ from metric definitions")
                for key, value in values.items():
                    cell_count += 1
                    label = f"task {task_id}/{model}/{key}"
                    if not isinstance(value, dict) or not self.finite_number(value.get("mean")):
                        self.error(f"{label}: mean must be finite")
                        continue
                    sd = value.get("sd")
                    if sd is not None and (not self.finite_number(sd) or sd < 0):
                        self.error(f"{label}: SD must be null or a finite nonnegative number")
            if len(set(models)) != len(models):
                self.error(f"task {task_id}: duplicate model rows")
        if row_count != 66 or cell_count != 238:
            self.error(f"benchmark inventory changed: expected 66 rows/238 metric cells; got {row_count}/{cell_count}")
        self.stats.update({"tasks": len(tasks), "model rows": row_count, "metric cells": cell_count})

    def image_manifest(self) -> None:
        manifest = self.read_json(REPO / "asset-manifest.json")
        if not isinstance(manifest, dict) or not isinstance(manifest.get("assets"), list):
            self.error("asset-manifest.json: expected assets[]")
            return
        seen = set()
        verified = 0
        for item in manifest["assets"]:
            if not isinstance(item, dict) or not isinstance(item.get("website"), str):
                self.error("asset-manifest.json: invalid asset entry")
                continue
            rel = item["website"]
            if rel in seen:
                self.error(f"asset-manifest.json: duplicate asset {rel}")
            seen.add(rel)
            path = (REPO / rel).resolve()
            if not self.within(path, DOCS.resolve()) or not path.is_file():
                self.error(f"asset-manifest.json: missing or non-public asset {rel}")
                continue
            expected = item.get("sha256")
            if not isinstance(expected, str) or re.fullmatch(r"[a-fA-F0-9]{64}", expected) is None:
                self.error(f"asset-manifest.json: invalid SHA-256 for {rel}")
                continue
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                header = stream.read(24)
                digest.update(header)
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            if digest.hexdigest().lower() != expected.lower():
                self.error(f"{rel}: SHA-256 differs from copied-paper-image manifest")
                continue
            if "bytes" in item and path.stat().st_size != item["bytes"]:
                self.error(f"{rel}: byte size differs from manifest")
            if header.startswith(b"\x89PNG\r\n\x1a\n") and len(header) == 24:
                width, height = struct.unpack(">II", header[16:24])
                if width != item.get("width") or height != item.get("height"):
                    self.error(f"{rel}: PNG dimensions differ from manifest")
            verified += 1
        images = DOCS / "assets" / "images"
        if images.is_dir():
            actual = {path.relative_to(REPO).as_posix() for path in images.iterdir() if path.is_file()}
            if actual != seen:
                self.error(f"image manifest coverage differs: unlisted={sorted(actual - seen)}, missing={sorted(seen - actual)}")
        self.stats["SHA-256-verified paper images"] = verified

    def run(self) -> int:
        self.files_and_encoding()
        self.html_documents()
        self.local_links()
        self.benchmark_json()
        self.image_manifest()
        if self.errors:
            print(f"FAIL: {len(self.errors)} website validation issue(s)")
            for error in self.errors:
                print(f"  - {error}")
            return 1
        print("PASS: cigAMT static website checks")
        for label, count in self.stats.items():
            print(f"  {label}: {count}")
        print("  Public docs/: no scientific arrays, model weights, local paths, or private provenance")
        print("  Mode: read-only; original research directories and external URLs were not accessed")
        return 0


def main() -> int:
    try:
        return Checks().run()
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f"FAIL: validation could not complete ({exc})")
        return 1


if __name__ == "__main__":
    sys.exit(main())

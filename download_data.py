#!/usr/bin/env python3
"""
Download ESI data files from Zenodo.

Usage:
  python3 download_data.py              # Download lookup database only (~650MB)
  python3 download_data.py --all        # Download lookup/raw data plus function catalogues (~1.9GB)
  python3 download_data.py --functions  # Download function catalogues only (for reproducing from scratch)

The lookup database (results pkl files) is all you need to use esi_integrate.py.
The unique-equation function catalogues are only needed to rerun the ESI
pipeline from its enumerated function inputs.
"""

import argparse
import hashlib
import os
import tarfile
import urllib.request
import shutil

# ── Zenodo record ───────────────────────────────────────────────────
ZENODO_RECORD = "20027938"
ZENODO_DOI = "10.5281/zenodo.20027938"
ZENODO_BASE = f"https://zenodo.org/records/{ZENODO_RECORD}/files"
ARCHIVE_NAME = "Archive.tar.gz"
ARCHIVE_URL = f"{ZENODO_BASE}/{ARCHIVE_NAME}?download=1"

# ── File manifest ──────────────────────────────────────────────────

# Lookup database: results pkl files (needed for esi_integrate.py)
LOOKUP_FILES = [
    "results/results_core_maths_k10.pkl",
    "results/results_core_log_maths_k8.pkl",
    "results/results_ext_maths_k8.pkl",
    "results/results_ext_maths_k9.pkl",
    "results/results_ext_log_maths_k8.pkl",
    "results/results_trig_maths_k8.pkl",
    "results/results_trig_maths_k9.pkl",
    "results/esi_hash_index.json",
]

# Raw results (needed for rho decomposition analysis)
RAW_FILES = [
    "results/raw_results_core_maths_k10.pkl",
    "results/raw_results_core_log_maths_k8.pkl",
    "results/raw_results_ext_maths_k9.pkl",
    "results/raw_results_ext_log_maths_k8.pkl",
    "results/raw_results_ext_log_maths_k9.pkl",
    "results/raw_results_trig_maths_k8.pkl",
    "results/raw_results_trig_maths_k9.pkl",
]

FUNCTION_ARCHIVE = "function_catalogues_unique_equations.tar.gz"


def download_file(url, dest, desc=None):
    """Download a file with progress."""
    if os.path.exists(dest):
        print(f"  Already exists: {dest}")
        return True

    os.makedirs(os.path.dirname(dest) or '.', exist_ok=True)
    desc = desc or os.path.basename(dest)

    try:
        print(f"  Downloading {desc}...", end=' ', flush=True)
        urllib.request.urlretrieve(url, dest)
        size = os.path.getsize(dest) / 1e6
        print(f"done ({size:.1f} MB)")
        return True
    except Exception as e:
        print(f"FAILED: {e}")
        return False


def download_archive(data_dir):
    """Download the published Zenodo archive."""
    archive = os.path.join(data_dir, ARCHIVE_NAME)
    download_file(ARCHIVE_URL, archive, desc=ARCHIVE_NAME)
    return archive


def _find_member(tar, wanted):
    """Find a path in the archive, allowing for one enclosing directory."""
    wanted = wanted.strip("/")
    matches = []
    for member in tar.getmembers():
        name = member.name.lstrip("./")
        if name == wanted or name.endswith("/" + wanted):
            matches.append(member)
    return matches[0] if len(matches) == 1 else None


def extract_files_from_archive(archive, data_dir, wanted_paths):
    """Extract selected files from the Zenodo archive into their repo paths."""
    with tarfile.open(archive, "r:gz") as tar:
        for wanted in wanted_paths:
            dest = os.path.join(data_dir, wanted)
            if os.path.exists(dest):
                print(f"  Already exists: {dest}")
                continue

            member = _find_member(tar, wanted)
            if member is None:
                print(f"  WARNING: {wanted} not found in {ARCHIVE_NAME}")
                continue
            if not member.isfile():
                print(f"  WARNING: {wanted} is not a regular file")
                continue

            os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
            source = tar.extractfile(member)
            if source is None:
                print(f"  WARNING: could not read {wanted}")
                continue
            with source, open(dest, "wb") as out:
                shutil.copyfileobj(source, out)
            print(f"  Extracted {wanted}")


def download_lookup(data_dir):
    """Download the lookup database (results pkl files)."""
    print("Downloading lookup database (needed for esi_integrate.py)...")
    archive = download_archive(data_dir)
    extract_files_from_archive(archive, data_dir, LOOKUP_FILES)


def download_raw(data_dir):
    """Download raw results (needed for decomposition analysis)."""
    print("\nDownloading raw results (needed for rho decomposition)...")
    archive = download_archive(data_dir)
    extract_files_from_archive(archive, data_dir, RAW_FILES)


def download_functions(data_dir):
    """Download unique-equation catalogues for reproducing the pipeline inputs."""
    print("\nDownloading function catalogues...")
    zenodo_archive = download_archive(data_dir)
    extract_files_from_archive(zenodo_archive, data_dir, [FUNCTION_ARCHIVE])
    function_archive = os.path.join(data_dir, FUNCTION_ARCHIVE)
    if os.path.exists(function_archive):
        with tarfile.open(function_archive, "r:gz") as tar:
            tar.extractall(data_dir)
        print(f"  Extracted function_catalogues/ into {data_dir}")


def main():
    parser = argparse.ArgumentParser(description="Download ESI data from Zenodo")
    parser.add_argument('--all', action='store_true',
                        help='Download everything (lookup db + raw results + function catalogues)')
    parser.add_argument('--functions', action='store_true',
                        help='Download function catalogues only (for reproducing from scratch)')
    parser.add_argument('--raw', action='store_true',
                        help='Download raw results (for decomposition analysis)')
    parser.add_argument('--dir', default='.',
                        help='Directory to download into (default: current)')
    args = parser.parse_args()

    if args.all:
        download_lookup(args.dir)
        download_raw(args.dir)
        download_functions(args.dir)
    elif args.functions:
        download_functions(args.dir)
    elif args.raw:
        download_lookup(args.dir)
        download_raw(args.dir)
    else:
        download_lookup(args.dir)

    print("\nDone! To use the lookup tool:")
    print(f"  python3 esi_integrate.py --data-dir {args.dir}/results \"1/x\"")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Download ESI data files from Zenodo.

Usage:
  python3 download_data.py              # Download lookup database only (~650MB)
  python3 download_data.py --all        # Download everything including function sets (~1.9GB)
  python3 download_data.py --functions  # Download function sets only (for reproducing from scratch)

The lookup database (results pkl files) is all you need to use esi_integrate.py.
To reproduce the pipeline from scratch, you also need the unique-equation
function catalogues.
"""

import argparse
import hashlib
import os
import sys
import tarfile
import urllib.request

# ── Zenodo record ───────────────────────────────────────────────────
# Update these after uploading to Zenodo
ZENODO_RECORD = "XXXXXXX"  # TODO: fill in after upload
ZENODO_BASE = f"https://zenodo.org/records/{ZENODO_RECORD}/files"

# ── File manifest ──────────────────────────────────────────────────

# Lookup database: results pkl files (needed for esi_integrate.py)
LOOKUP_FILES = [
    ("results/results_core_maths_k10.pkl", "results_core_maths_k10.pkl"),
    ("results/results_core_log_maths_k8.pkl", "results_core_log_maths_k8.pkl"),
    ("results/results_ext_maths_k8.pkl", "results_ext_maths_k8.pkl"),
    ("results/results_ext_maths_k9.pkl", "results_ext_maths_k9.pkl"),
    ("results/results_ext_log_maths_k8.pkl", "results_ext_log_maths_k8.pkl"),
    ("results/results_trig_maths_k8.pkl", "results_trig_maths_k8.pkl"),
    ("results/results_trig_maths_k9.pkl", "results_trig_maths_k9.pkl"),
    ("results/esi_hash_index.json", "esi_hash_index.json"),
]

# Raw results (needed for rho decomposition analysis)
RAW_FILES = [
    ("results/raw_results_core_maths_k10.pkl", "raw_results_core_maths_k10.pkl"),
    ("results/raw_results_core_log_maths_k8.pkl", "raw_results_core_log_maths_k8.pkl"),
    ("results/raw_results_ext_maths_k9.pkl", "raw_results_ext_maths_k9.pkl"),
    ("results/raw_results_ext_log_maths_k8.pkl", "raw_results_ext_log_maths_k8.pkl"),
    ("results/raw_results_ext_log_maths_k9.pkl", "raw_results_ext_log_maths_k9.pkl"),
    ("results/raw_results_trig_maths_k8.pkl", "raw_results_trig_maths_k8.pkl"),
    ("results/raw_results_trig_maths_k9.pkl", "raw_results_trig_maths_k9.pkl"),
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


def download_lookup(data_dir):
    """Download the lookup database (results pkl files)."""
    print("Downloading lookup database (needed for esi_integrate.py)...")
    results_dir = os.path.join(data_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    for local_path, remote_name in LOOKUP_FILES:
        dest = os.path.join(data_dir, local_path)
        url = f"{ZENODO_BASE}/{remote_name}"
        download_file(url, dest)


def download_raw(data_dir):
    """Download raw results (needed for decomposition analysis)."""
    print("\nDownloading raw results (needed for rho decomposition)...")
    results_dir = os.path.join(data_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    for local_path, remote_name in RAW_FILES:
        dest = os.path.join(data_dir, local_path)
        url = f"{ZENODO_BASE}/{remote_name}"
        download_file(url, dest)


def download_functions(data_dir):
    """Download unique-equation catalogues for reproducing the pipeline inputs."""
    print("\nDownloading function sets...")
    archive = os.path.join(data_dir, FUNCTION_ARCHIVE)
    url = f"{ZENODO_BASE}/{FUNCTION_ARCHIVE}"
    if download_file(url, archive, desc=FUNCTION_ARCHIVE):
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(data_dir)
        print(f"  Extracted function_catalogues/ into {data_dir}")


def main():
    parser = argparse.ArgumentParser(description="Download ESI data from Zenodo")
    parser.add_argument('--all', action='store_true',
                        help='Download everything (lookup db + raw results + function sets)')
    parser.add_argument('--functions', action='store_true',
                        help='Download function sets only (for reproducing from scratch)')
    parser.add_argument('--raw', action='store_true',
                        help='Download raw results (for decomposition analysis)')
    parser.add_argument('--dir', default='.',
                        help='Directory to download into (default: current)')
    args = parser.parse_args()

    if ZENODO_RECORD == "XXXXXXX":
        print("NOTE: Zenodo record IDs not yet filled in.")
        print("This script will work once the data is uploaded to Zenodo")
        print("and the record IDs are updated at the top of this file.")
        print()
        print("For now, contact the authors for data access,")
        print("or run the ESI pipeline to generate the data yourself.")
        print("See README.md for instructions.")
        sys.exit(0)

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

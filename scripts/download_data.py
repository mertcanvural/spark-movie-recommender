#!/usr/bin/env python3
"""
Download the MovieLens dataset and extract it to the raw data directory.

Usage:
  python scripts/download_data.py              # Downloads all datasets
  python scripts/download_data.py 10m          # Downloads 10m dataset
  python scripts/download_data.py benchmark    # Downloads all datasets (100k, 1m, 10m, 25m)
  python scripts/download_data.py all          # Same as benchmark, downloads all datasets
"""

import os
import sys
import zipfile
import requests
from pathlib import Path

# URLs for different MovieLens datasets
MOVIELENS_URLS = {
    "100k": "https://files.grouplens.org/datasets/movielens/ml-100k.zip",
    "1m": "https://files.grouplens.org/datasets/movielens/ml-1m.zip",
    "10m": "https://files.grouplens.org/datasets/movielens/ml-10m.zip",
    "25m": "https://files.grouplens.org/datasets/movielens/ml-25m.zip",
}


def download_file(url, destination):
    """Download a file from a URL to a destination path."""
    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024  # 1 Kibibyte
    downloaded = 0

    print(f"Downloading {url} to {destination}")
    with open(destination, "wb") as file:
        for data in response.iter_content(block_size):
            downloaded += len(data)
            file.write(data)

            # display progress
            done = int(50 * downloaded / total_size)
            progress = f"\r[{'=' * done}{' ' * (50 - done)}]"
            sys.stdout.write(f"{progress} {downloaded}/{total_size} bytes")
            sys.stdout.flush()
    print("\nDownload complete!")


def extract_zip(zip_path, extract_to):
    """Extract a zip file to a destination directory."""
    print(f"Extracting {zip_path} to {extract_to}")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)
    print("Extraction complete!")


def download_dataset(size, raw_data_dir, keep_zip=False):
    """Download and extract a specific dataset size."""
    if size not in MOVIELENS_URLS:
        print(f"Error: Unknown dataset size '{size}'")
        return False

    print(f"Processing MovieLens {size} dataset")

    # construct file paths
    url = MOVIELENS_URLS[size]
    zip_filename = f"ml-{size}.zip"
    zip_path = raw_data_dir / zip_filename
    extract_dir = raw_data_dir / f"ml-{size}"

    # Skip download if already extracted
    if extract_dir.exists():
        print(f"Dataset {size} already exists at {extract_dir}")
        return True

    # Download if zip doesn't exist
    if not zip_path.exists():
        try:
            download_file(url, zip_path)
        except Exception as e:
            print(f"Error downloading {size} dataset: {e}")
            return False
    else:
        print(f"Zip file already exists at {zip_path}")

    # Extract
    try:
        extract_zip(zip_path, raw_data_dir)
    except Exception as e:
        print(f"Error extracting {size} dataset: {e}")
        return False

    # Delete zip file if requested
    if not keep_zip and zip_path.exists():
        os.remove(zip_path)
        print(f"Deleted {zip_path}")

    return True


def main():
    # create the data/raw directory if it doesn't exist
    raw_data_dir = Path("data/raw")
    raw_data_dir.mkdir(parents=True, exist_ok=True)

    # By default, download all datasets
    download_all = True
    keep_zip = False
    sizes_to_download = list(MOVIELENS_URLS.keys())

    if len(sys.argv) > 1:
        if sys.argv[1] == "benchmark" or sys.argv[1] == "all":
            # Already set to download all
            pass
        elif sys.argv[1] in MOVIELENS_URLS:
            # Download specific size only
            sizes_to_download = [sys.argv[1]]
            download_all = False

    # Download datasets
    if download_all:
        print(f"Downloading all MovieLens datasets: {', '.join(sizes_to_download)}")

    success_count = 0
    for size in sizes_to_download:
        if download_dataset(size, raw_data_dir, keep_zip):
            success_count += 1

    print(f"Downloaded {success_count} out of {len(sizes_to_download)} datasets")


if __name__ == "__main__":
    main()

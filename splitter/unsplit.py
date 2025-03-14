#!/usr/bin/env python3

import os
import sys
import tarfile
import glob
import re
from pathlib import Path


def unsplit_file(original_filename):
    """
    Reassembles a file that was split using split.py

    Args:
        original_filename: The original filename without chunk numbers
    """
    original_path = Path(original_filename)

    # Extract stem and extension
    name = original_path.stem
    ext = original_path.suffix
    directory = original_path.parent

    # Find all chunks matching the pattern: name + digit + extension
    pattern = str(directory / f"{name}[0-9]*{ext}")
    chunks = glob.glob(pattern)

    if not chunks:
        print(f"Error: No chunks found matching pattern '{pattern}'")
        return False

    # Sort chunks numerically (name0.ext, name1.ext, etc.)
    def chunk_number(chunk_path):
        match = re.search(f"{name}([0-9]+){ext}$", os.path.basename(chunk_path))
        return int(match.group(1)) if match else -1

    chunks.sort(key=chunk_number)

    # Temporary compressed file
    temp_tar_path = directory / f"{name}_reconstructed.tar.gz"

    try:
        print(f"Reassembling {len(chunks)} chunks into {temp_tar_path}...")

        # Combine all chunks
        with open(temp_tar_path, "wb") as outfile:
            for i, chunk_path in enumerate(chunks):
                print(f"Processing chunk {i+1}/{len(chunks)}: {chunk_path}")
                with open(chunk_path, "rb") as infile:
                    outfile.write(infile.read())

        print("Extracting the original file from the compressed archive...")

        # Extract the original file
        with tarfile.open(temp_tar_path, "r:gz") as tar:
            members = tar.getmembers()

            if not members:
                print("Error: No files found in the tar archive.")
                return False

            # Extract to original location
            tar.extractall(path=directory)

            # Get the name of the extracted file(s)
            extracted_files = [member.name for member in members]
            print(f"Successfully extracted: {', '.join(extracted_files)}")

        return True

    except Exception as e:
        print(f"Error during file reassembly: {e}")
        return False

    finally:
        # Clean up the temporary tar file
        if temp_tar_path.exists():
            os.remove(temp_tar_path)
            print(f"Removed temporary compressed file: {temp_tar_path}")


def main():
    if len(sys.argv) != 2:
        print("Usage: python unsplit.py <original_filename>")
        print("Note: Provide the original filename without chunk numbers")
        sys.exit(1)

    original_filename = sys.argv[1]
    unsplit_file(original_filename)


if __name__ == "__main__":
    main()

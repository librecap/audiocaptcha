#!/usr/bin/env python3

import os
import sys
import tarfile
import math
from pathlib import Path


def split_file(file_path, chunk_size_bytes=90_000_000):
    """
    Compresses a file using tar with highest compression and splits it into chunks under 90 Mbit.

    Args:
        file_path: Path to the file to be split
        chunk_size_bytes: Size of each chunk in bytes (default: 90 Mbit)
    """
    file_path = Path(file_path)

    if not file_path.exists():
        print(f"Error: File '{file_path}' does not exist.")
        return False

    # Create a temporary compressed tar file
    temp_tar_path = file_path.with_suffix(".tar.gz")

    try:
        print(f"Compressing {file_path} with highest compression level...")
        with tarfile.open(temp_tar_path, "w:gz", compresslevel=9) as tar:
            tar.add(file_path, arcname=file_path.name)

        # Get the size of the compressed file
        compressed_size = os.path.getsize(temp_tar_path)

        # Calculate the number of chunks needed
        num_chunks = math.ceil(compressed_size / chunk_size_bytes)

        print(
            f"Splitting into {num_chunks} chunk(s) of {chunk_size_bytes/1_000_000} Mbit each..."
        )

        # Get the original file name without extension and with extension
        original_name = file_path.stem
        original_ext = file_path.suffix
        output_dir = file_path.parent

        # Read and split the compressed file
        with open(temp_tar_path, "rb") as f:
            for i in range(num_chunks):
                chunk_data = f.read(chunk_size_bytes)
                if not chunk_data:
                    break

                # Create the output file name: original0.ext, original1.ext, etc.
                output_path = output_dir / f"{original_name}{i}{original_ext}"

                with open(output_path, "wb") as chunk_file:
                    chunk_file.write(chunk_data)

                print(f"Created chunk {i+1}/{num_chunks}: {output_path}")

        print(f"Successfully split {file_path} into {num_chunks} chunks.")
        return True

    except Exception as e:
        print(f"Error while splitting file: {e}")
        return False

    finally:
        # Clean up the temporary tar file
        if temp_tar_path.exists():
            os.remove(temp_tar_path)
            print(f"Removed temporary compressed file: {temp_tar_path}")


def main():
    if len(sys.argv) != 2:
        print("Usage: python split.py <file_path>")
        sys.exit(1)

    file_path = sys.argv[1]
    split_file(file_path)


if __name__ == "__main__":
    main()

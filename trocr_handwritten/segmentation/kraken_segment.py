#!/usr/bin/env python
"""
Kraken-based line segmentation using subprocess to call kraken from a separate environment.

This module wraps kraken CLI since kraken requires Python ≤3.11 and torch <2.5,
which is incompatible with this project's dependencies.

Usage:
    python -m trocr_handwritten.segmentation.kraken_segment -i image.jpg -o output/
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

import cv2


def check_kraken_available(conda_env: str = "kraken_env") -> bool:
    """Check if kraken is available in the specified conda environment."""
    try:
        result = subprocess.run(
            f"conda run -n {conda_env} kraken --version",
            shell=True,
            capture_output=True,
            text=True,
        )
        return result.returncode == 0
    except Exception:
        return False


def segment_with_kraken(
    image_path: str,
    output_dir: str,
    conda_env: str = "kraken_env",
    model: Optional[str] = None,
    padding: int = 5,
) -> int:
    """
    Segment an image into lines using kraken via subprocess.

    Args:
        image_path: Path to input image
        output_dir: Directory to save line crops
        conda_env: Name of conda environment with kraken installed
        model: Path to custom segmentation model (optional)
        padding: Padding around detected lines

    Returns:
        Number of lines extracted
    """
    image_path = Path(image_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Temporary JSON file for kraken output
    json_output = output_dir / "kraken_segments.json"

    # Build kraken command
    cmd = f"conda run -n {conda_env} kraken -i {image_path} segment -bl -o {json_output}"
    if model:
        cmd += f" -m {model}"

    print(f"Running: {cmd}")

    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=120,
        )

        if result.returncode != 0:
            print(f"Kraken error: {result.stderr}")
            raise RuntimeError(f"Kraken segmentation failed: {result.stderr}")

    except subprocess.TimeoutExpired:
        raise RuntimeError("Kraken segmentation timed out")

    # Parse kraken output and extract line crops
    if not json_output.exists():
        # Try alternative output format (kraken sometimes outputs differently)
        alt_output = image_path.with_suffix(".json")
        if alt_output.exists():
            json_output = alt_output
        else:
            raise RuntimeError(f"Kraken output not found: {json_output}")

    # Load image for cropping
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    height, width = img.shape[:2]

    # Parse JSON output
    with open(json_output, "r") as f:
        data = json.load(f)

    # Extract lines from kraken output
    lines = data.get("lines", [])
    if not lines and "boxes" in data:
        # Alternative format
        lines = [{"boundary": box} for box in data["boxes"]]

    count = 0
    for idx, line in enumerate(lines):
        # Get bounding box from boundary polygon
        boundary = line.get("boundary", [])
        if not boundary:
            baseline = line.get("baseline", [])
            if baseline:
                # Estimate boundary from baseline
                xs = [p[0] for p in baseline]
                ys = [p[1] for p in baseline]
                y_center = sum(ys) / len(ys)
                line_height = 50  # Estimated
                boundary = [
                    [min(xs), y_center - line_height * 0.7],
                    [max(xs), y_center - line_height * 0.7],
                    [max(xs), y_center + line_height * 0.3],
                    [min(xs), y_center + line_height * 0.3],
                ]
            else:
                continue

        # Get bounding box
        xs = [p[0] for p in boundary]
        ys = [p[1] for p in boundary]
        x1, x2 = int(min(xs)), int(max(xs))
        y1, y2 = int(min(ys)), int(max(ys))

        # Add padding
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(width, x2 + padding)
        y2 = min(height, y2 + padding)

        # Crop and save
        if x2 > x1 and y2 > y1:
            line_crop = img[y1:y2, x1:x2]
            if line_crop.size > 0:
                save_path = output_dir / f"line_{count:03d}.jpg"
                cv2.imwrite(str(save_path), line_crop)
                count += 1

    # Clean up temp file
    if json_output.exists() and json_output.name == "kraken_segments.json":
        json_output.unlink()

    return count


def main():
    parser = argparse.ArgumentParser(
        description="Segment document images into text lines using Kraken"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input image file or directory containing images"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output directory for line crops"
    )
    parser.add_argument(
        "--conda-env",
        default="kraken_env",
        help="Conda environment with kraken installed (default: kraken_env)"
    )
    parser.add_argument(
        "--model", "-m",
        default=None,
        help="Path to custom kraken segmentation model"
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=5,
        help="Padding around detected lines"
    )
    parser.add_argument(
        "--recursive", "-r",
        action="store_true",
        help="Process directories recursively"
    )

    args = parser.parse_args()

    # Check kraken availability
    if not check_kraken_available(args.conda_env):
        print(f"Error: Kraken not found in conda environment '{args.conda_env}'")
        print("Install kraken with:")
        print(f"  bash trocr_handwritten/segmentation/install_kraken.sh")
        print("Or manually:")
        print(f"  conda create -n {args.conda_env} python=3.11 -y")
        print(f"  conda activate {args.conda_env}")
        print("  pip install kraken")
        sys.exit(1)

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"Error: Input path does not exist: {input_path}")
        sys.exit(1)

    # Collect images to process
    if input_path.is_file():
        images = [input_path]
    else:
        pattern = "**/*.jpg" if args.recursive else "*.jpg"
        images = list(input_path.glob(pattern))
        pattern_png = "**/*.png" if args.recursive else "*.png"
        images.extend(input_path.glob(pattern_png))

    if not images:
        print(f"No images found in {input_path}")
        sys.exit(1)

    print(f"Found {len(images)} image(s) to process")
    print(f"Using Kraken from conda env: {args.conda_env}")

    total_lines = 0
    for img_path in images:
        # Create output subdirectory for each image
        if input_path.is_dir():
            rel_path = img_path.relative_to(input_path)
            img_output_dir = output_path / rel_path.stem / "lines"
        else:
            img_output_dir = output_path / "lines"

        try:
            count = segment_with_kraken(
                str(img_path),
                str(img_output_dir),
                conda_env=args.conda_env,
                model=args.model,
                padding=args.padding,
            )
            print(f"  {img_path.name}: {count} lines")
            total_lines += count
        except Exception as e:
            print(f"  {img_path.name}: ERROR - {e}")

    print(f"\nTotal: {total_lines} lines extracted")


if __name__ == "__main__":
    main()

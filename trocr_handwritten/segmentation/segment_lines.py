#!/usr/bin/env python
"""
Command-line interface for line segmentation.

Usage:
    python -m trocr_handwritten.segmentation.segment_lines --input image.jpg --output lines/
    segment-lines -i image.jpg -o output/
"""

import argparse
import os
import sys
from pathlib import Path

from .line_segmenter import segment_lines, ProjectionSegmenter


def main():
    parser = argparse.ArgumentParser(
        description="Segment document images into text lines"
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
        # Also check for png
        pattern_png = "**/*.png" if args.recursive else "*.png"
        images.extend(input_path.glob(pattern_png))

    if not images:
        print(f"No images found in {input_path}")
        sys.exit(1)

    print(f"Found {len(images)} image(s) to process")
    print("Using method: projection")

    # Create segmenter
    segmenter = ProjectionSegmenter(padding=args.padding)

    total_lines = 0
    for img_path in images:
        # Create output subdirectory for each image
        if input_path.is_dir():
            rel_path = img_path.relative_to(input_path)
            img_output_dir = output_path / rel_path.stem / "lines"
        else:
            img_output_dir = output_path / "lines"

        try:
            count = segmenter.segment(str(img_path), str(img_output_dir))
            print(f"  {img_path.name}: {count} lines")
            total_lines += count
        except Exception as e:
            print(f"  {img_path.name}: ERROR - {e}")

    print(f"\nTotal: {total_lines} lines extracted")


if __name__ == "__main__":
    main()

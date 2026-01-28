"""
Line segmentation module using projection-based methods.
Detects black text lines separated by white gaps.
"""

import os
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional

import cv2
import numpy as np


class LineSegmenter(ABC):
    """Abstract base class for line segmentation."""

    @abstractmethod
    def segment(self, image_path: str, output_dir: str) -> int:
        """
        Segment an image into lines and save them.

        Args:
            image_path: Path to the input image
            output_dir: Directory to save line crops

        Returns:
            Number of lines extracted
        """
        pass


def remove_black_borders(img: np.ndarray, threshold: int = 30) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """
    Remove black borders from an image and return the cropped content.
    
    Args:
        img: Input image (BGR or grayscale)
        threshold: Pixel value below which is considered black border
        
    Returns:
        Tuple of (cropped_image, (x1, y1, x2, y2) bounding box)
    """
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    # Find rows and columns that have content (not just black)
    row_has_content = np.max(gray, axis=1) > threshold
    col_has_content = np.max(gray, axis=0) > threshold
    
    # Find bounding box of content
    rows_with_content = np.where(row_has_content)[0]
    cols_with_content = np.where(col_has_content)[0]
    
    if len(rows_with_content) == 0 or len(cols_with_content) == 0:
        # No content found, return original
        return img, (0, 0, img.shape[1], img.shape[0])
    
    y1, y2 = rows_with_content[0], rows_with_content[-1] + 1
    x1, x2 = cols_with_content[0], cols_with_content[-1] + 1
    
    # Add small margin
    margin = 5
    y1 = max(0, y1 - margin)
    y2 = min(img.shape[0], y2 + margin)
    x1 = max(0, x1 - margin)
    x2 = min(img.shape[1], x2 + margin)
    
    return img[y1:y2, x1:x2], (x1, y1, x2, y2)


def split_two_pages(img: np.ndarray, min_aspect_ratio: float = 1.2) -> List[np.ndarray]:
    """
    Split a 2-page spread into separate left and right pages.
    
    Args:
        img: Input image (BGR)
        min_aspect_ratio: Minimum width/height ratio to consider as 2-page spread
        
    Returns:
        List of page images [left_page, right_page] or [original] if not a spread
    """
    height, width = img.shape[:2]
    aspect_ratio = width / height
    
    # Only split if image is wide enough to be a 2-page spread
    if aspect_ratio < min_aspect_ratio:
        return [img]
    
    # Convert to grayscale for analysis
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    # Find the center gap (vertical line with minimal content)
    # Look in the middle 20% of the image
    center_start = int(width * 0.4)
    center_end = int(width * 0.6)
    
    # Vertical projection in center region
    center_region = gray[:, center_start:center_end]
    vertical_proj = np.sum(center_region, axis=0)
    
    # Invert (we want minimum content = binding area)
    # Find the column with minimum projection (darkest vertical line = gap)
    min_idx = np.argmin(vertical_proj)
    split_x = center_start + min_idx
    
    # Split into two pages
    left_page = img[:, :split_x]
    right_page = img[:, split_x:]
    
    return [left_page, right_page]


class ProjectionSegmenter(LineSegmenter):
    """
    Line segmentation using horizontal projection profile.
    Detects black text lines separated by white gaps.
    """

    def __init__(
        self,
        min_line_height: int = 30,
        max_line_height: int = 150,
        padding: int = 5,
        gap_threshold: float = 0.15,
        kernel_size: int = 3,
        remove_borders: bool = True,
        split_pages: bool = True,
    ):
        """
        Args:
            min_line_height: Minimum pixel height for a valid line
            max_line_height: Maximum pixel height (larger regions are split)
            padding: Vertical padding to add around each line
            gap_threshold: Normalized projection value below which is considered a gap
            kernel_size: Smoothing kernel size for projection
            remove_borders: Remove black borders before processing
            split_pages: Split 2-page spreads into separate pages
        """
        self.min_line_height = min_line_height
        self.max_line_height = max_line_height
        self.padding = padding
        self.gap_threshold = gap_threshold
        self.kernel_size = kernel_size
        self.remove_borders = remove_borders
        self.split_pages = split_pages

    def segment(self, image_path: str, output_dir: str) -> int:
        """Segment image into lines using horizontal projection."""
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")

        # Remove black borders first
        if self.remove_borders:
            img, bbox = remove_black_borders(img)

        # Split into pages if it's a 2-page spread
        if self.split_pages:
            pages = split_two_pages(img)
        else:
            pages = [img]

        os.makedirs(output_dir, exist_ok=True)
        
        total_lines = 0
        for page_idx, page_img in enumerate(pages):
            # Remove borders from each page individually
            if self.remove_borders and len(pages) > 1:
                page_img, _ = remove_black_borders(page_img)
            
            lines = self._segment_page(page_img)
            
            # Save with page prefix if multiple pages
            prefix = f"p{page_idx}_" if len(pages) > 1 else ""
            for line_idx, (y1, y2) in enumerate(lines):
                line_crop = page_img[y1:y2, :]
                save_path = os.path.join(output_dir, f"{prefix}line_{line_idx:03d}.jpg")
                cv2.imwrite(save_path, line_crop)
                total_lines += 1
        
        return total_lines

    def _segment_page(self, img: np.ndarray) -> List[Tuple[int, int]]:
        """Segment a single page into lines."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Binarize using adaptive thresholding
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 21, 10
        )

        # Horizontal projection (sum of white/text pixels per row)
        projection = np.sum(binary, axis=1).astype(float)

        # Normalize projection to 0-1 range
        proj_min = projection.min()
        proj_max = projection.max()
        if proj_max > proj_min:
            normalized = (projection - proj_min) / (proj_max - proj_min)
        else:
            normalized = np.zeros_like(projection)

        # Light smoothing
        kernel = np.ones(self.kernel_size) / self.kernel_size
        smoothed = np.convolve(normalized, kernel, mode="same")

        # Find gaps (where projection drops significantly)
        is_gap = smoothed < self.gap_threshold

        return self._find_lines_from_gaps(is_gap, smoothed, img.shape[0])

    def _find_lines_from_gaps(
        self, is_gap: np.ndarray, projection: np.ndarray, height: int
    ) -> List[Tuple[int, int]]:
        """Find line boundaries by detecting gaps between text."""
        lines = []
        start = None

        for y in range(height):
            if not is_gap[y]:  # In text region
                if start is None:
                    start = y
            else:  # In gap
                if start is not None:
                    end = y
                    line_height = end - start
                    if line_height >= self.min_line_height:
                        # If line is too tall, try to split it
                        if line_height > self.max_line_height:
                            sub_lines = self._split_tall_region(
                                start, end, projection
                            )
                            lines.extend(sub_lines)
                        else:
                            y1 = max(0, start - self.padding)
                            y2 = min(height, end + self.padding)
                            lines.append((y1, y2))
                    start = None

        # Handle last region
        if start is not None:
            end = height
            line_height = end - start
            if line_height >= self.min_line_height:
                if line_height > self.max_line_height:
                    sub_lines = self._split_tall_region(start, end, projection)
                    lines.extend(sub_lines)
                else:
                    lines.append((max(0, start - self.padding), end))

        return lines

    def _split_tall_region(
        self, start: int, end: int, projection: np.ndarray
    ) -> List[Tuple[int, int]]:
        """Split a tall region into individual lines by finding local minima."""
        region_proj = projection[start:end]
        
        local_min = np.min(region_proj)
        local_max = np.max(region_proj)
        
        if local_max <= local_min:
            return [(start, end)]
        
        # Normalize within region
        region_norm = (region_proj - local_min) / (local_max - local_min)
        
        # Find valleys (local minima below 0.3)
        valley_threshold = 0.3
        is_valley = region_norm < valley_threshold
        
        # Find line boundaries within region
        sub_lines = []
        line_start = None
        
        for i in range(len(region_norm)):
            if not is_valley[i]:
                if line_start is None:
                    line_start = i
            else:
                if line_start is not None:
                    line_end = i
                    if line_end - line_start >= self.min_line_height // 2:
                        y1 = max(0, start + line_start - self.padding)
                        y2 = min(end, start + line_end + self.padding)
                        sub_lines.append((y1, y2))
                    line_start = None
        
        # Handle last sub-line
        if line_start is not None:
            line_end = len(region_norm)
            if line_end - line_start >= self.min_line_height // 2:
                y1 = max(0, start + line_start - self.padding)
                y2 = min(end, start + line_end + self.padding)
                sub_lines.append((y1, y2))
        
        return sub_lines if sub_lines else [(start, end)]

    def _save_line_crops(
        self, img: np.ndarray, lines: List[Tuple[int, int]], output_dir: str
    ) -> int:
        """Save line crops to output directory."""
        for idx, (y1, y2) in enumerate(lines):
            line_crop = img[y1:y2, :]
            save_path = os.path.join(output_dir, f"line_{idx:03d}.jpg")
            cv2.imwrite(save_path, line_crop)
        return len(lines)


def segment_lines(
    image_path: str,
    output_dir: str,
    **kwargs,
) -> int:
    """
    Convenience function to segment lines from an image.

    Args:
        image_path: Path to input image
        output_dir: Directory to save line crops
        **kwargs: Additional arguments passed to segmenter

    Returns:
        Number of lines extracted
    """
    segmenter = ProjectionSegmenter(**kwargs)
    return segmenter.segment(image_path, output_dir)

from trocr_handwritten.ner import ner_GPT
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, AutoTokenizer, GenerationConfig
from trocr_handwritten.trocr.apply_trocr import process_folder
from trocr_handwritten.utils.logging_config import get_logger
from trocr_handwritten.parse.utils import YOLOv10Model, create_structured_crops
from trocr_handwritten.parse.settings import LayoutParserSettings
from trocr_handwritten.segmentation import ProjectionSegmenter, segment_with_kraken, check_kraken_available
import os
import argparse
import sys
import shutil
import cv2
import numpy as np
from pathlib import Path
import json

# Import project modules
# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))


# NER Imports

logger = get_logger(__name__)


def split_lines_projection(image_path, output_dir):
    """
    Split text block into lines using horizontal projection profile.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        logger.error(f"Could not read image: {image_path}")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Binarize (Otsu's method)
    _, binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Horizontal projection
    projection = np.sum(binary, axis=1)

    # Smooth projection to avoid noise
    kernel_size = 5
    kernel = np.ones(kernel_size) / kernel_size
    smoothed = np.convolve(projection, kernel, mode='same')

    # Find valleys (gaps between lines)
    # Simple heuristic: if projection is below valid threshold, it's a gap
    threshold = np.max(smoothed) * 0.05
    is_space = smoothed < threshold

    lines = []
    start = None

    height = img.shape[0]

    for y in range(height):
        if not is_space[y]:
            if start is None:
                start = y
        else:
            if start is not None:
                # End of a line
                end = y
                if end - start > 10:  # Min height for a line
                    # Add some padding
                    y1 = max(0, start - 2)
                    y2 = min(height, end + 2)
                    lines.append((y1, y2))
                start = None

    # Handle last line
    if start is not None:
        end = height
        if end - start > 10:
            lines.append((max(0, start - 2), end))

    # Save crops
    os.makedirs(output_dir, exist_ok=True)
    count = 0
    for y1, y2 in lines:
        line_crop = img[y1:y2, :]
        save_path = os.path.join(output_dir, f"line_{count:03d}.jpg")
        cv2.imwrite(save_path, line_crop)
        count += 1

    return count


def main():
    parser = argparse.ArgumentParser(
        description="Run Full Pipeline: Parse -> OCR -> NER")
    parser.add_argument("image_path", help="Path to input image")
    parser.add_argument("--output_dir", default="data/output",
                        help="Directory for outputs")
    parser.add_argument("--groq_api_key", default=None,
                        help="Groq API Key (optional if env var set)")
    parser.add_argument("--cleanup", action="store_true",
                        help="Clean up intermediate files")
    parser.add_argument("--segmentation", choices=["projection", "kraken"],
                        default="projection",
                        help="Line segmentation method (default: projection)")
    parser.add_argument("--skip_layout", action="store_true",
                        help="Skip layout parsing, process entire image as text")
    parser.add_argument("--kraken_env", default="kraken_env",
                        help="Conda environment with kraken (default: kraken_env)")

    args = parser.parse_args()

    image_path = Path(args.image_path).resolve()
    if not image_path.exists():
        logger.error(f"Image not found: {image_path}")
        sys.exit(1)

    work_dir = Path(args.output_dir).resolve()
    image_work_dir = work_dir / "images" / image_path.stem

    # Select segmentation method
    use_kraken = args.segmentation == "kraken"
    if use_kraken:
        if not check_kraken_available(args.kraken_env):
            logger.error(f"Kraken not found in conda env '{args.kraken_env}'")
            logger.error("Install with: bash trocr_handwritten/segmentation/install_kraken.sh")
            sys.exit(1)
        logger.info(f"Using Kraken for line segmentation (env: {args.kraken_env})")
    else:
        segmenter = ProjectionSegmenter()
        logger.info("Using projection method for line segmentation")

    # Step 1: Layout Parsing (optional)
    # Note: Public model uses "plain text" class, private model used "Plein Texte"
    text_dir = image_work_dir / "plain text"  # For public DocStructBench model
    full_text_dir = image_work_dir / "Plein Texte"  # For private model (legacy)
    lines_dir = image_work_dir / "lines"

    if args.skip_layout:
        logger.info("Step 1: Layout Parsing SKIPPED (--skip_layout)")
        # Use entire image as text block
        os.makedirs(text_dir, exist_ok=True)
        shutil.copy(image_path, text_dir / "000.jpg")
    else:
        logger.info("Step 1: Layout Parsing")
        temp_input_dir = work_dir / "temp_input"
        os.makedirs(temp_input_dir, exist_ok=True)
        shutil.copy(image_path, temp_input_dir / image_path.name)

        settings = LayoutParserSettings(
            path_folder=str(temp_input_dir),
            path_output=str(work_dir / "images"),
            device="cpu"  # Or cuda if available
        )

        # Initialize Layout Model
        logger.info("Loading Layout Model...")
        try:
            model = YOLOv10Model(settings, logger)
            det_res = model.predict(settings.path_folder)
            create_structured_crops(
                det_res, settings.class_names, settings.path_output)
        except Exception as e:
            logger.error(f"Layout parsing failed: {e}")
            logger.warning(
                "Falling back to processing entire image as text block.")
            os.makedirs(text_dir, exist_ok=True)
            shutil.copy(image_path, text_dir / "fallback_000.jpg")

    # Find text directory (support both old and new class names)
    if text_dir.exists():
        source_dir = text_dir
    elif full_text_dir.exists():
        source_dir = full_text_dir
    else:
        source_dir = None

    # Step 2: Line Segmentation
    if source_dir and source_dir.exists():
        logger.info(f"Step 2: Line Segmentation (method: {args.segmentation})")
        total_lines = 0
        for crop_file in source_dir.glob("*.jpg"):
            logger.info(f"Segmenting lines for {crop_file}")
            if use_kraken:
                count = segment_with_kraken(
                    str(crop_file), str(lines_dir), conda_env=args.kraken_env
                )
            else:
                count = segmenter.segment(str(crop_file), str(lines_dir))
            total_lines += count
        logger.info(f"Extracted {total_lines} lines.")
    else:
        logger.error("No text regions found. Check layout parser output or use --skip_layout.")

    if not lines_dir.exists() or not list(lines_dir.glob("*.jpg")):
        logger.error("No lines generated. Aborting OCR.")
        sys.exit(1)

    # Step 3: OCR
    logger.info("Step 3: OCR")
    model_name = "microsoft/trocr-large-handwritten"

    # Login/Auth not handled here, assumed public model

    logger.info(f"Loading TrOCR model: {model_name}")
    processor = TrOCRProcessor.from_pretrained(model_name, use_fast=True)
    tokenizer = AutoTokenizer.from_pretrained("agomberto/trocr-large-handwritten-fr")
    trocr_model = VisionEncoderDecoderModel.from_pretrained("agomberto/trocr-large-handwritten-fr")
    generation_config = GenerationConfig.from_pretrained(model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trocr_model.to(device)

    logger.info("Running OCR on lines...")
    process_folder(
        str(lines_dir),
        trocr_model,
        processor,
        tokenizer,
        device,
        generation_config
    )

    transcription_file = image_work_dir / "transcriptions.txt"
    if not transcription_file.exists():
        logger.error("Transcription file not found.")
        sys.exit(1)

    # Read transcription
    with open(transcription_file, "r") as f:
        lines = f.readlines()

    full_text = " ".join([line.split("\t")[1].strip() for line in lines])
    logger.info(f"Transcription:\n{full_text}")

    # Save full text
    full_text_path = image_work_dir / "full_text.txt"
    with open(full_text_path, "w") as f:
        f.write(full_text)

    # Step 4: NER - SKIPPED as per user request
    logger.info("Step 4: NER - SKIPPED")
    # cmd = [ ... ]
    # subprocess.run( ... )

    if args.cleanup:
        shutil.rmtree(temp_input_dir)


if __name__ == "__main__":
    main()

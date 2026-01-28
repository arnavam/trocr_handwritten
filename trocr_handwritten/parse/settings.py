from dataclasses import dataclass


@dataclass
class LayoutParserSettings:
    path_folder: str = "data/raw/images"
    path_output: str = "data/processed/images/"
    path_model: str = None
    # Public DocLayout-YOLO model (DocStructBench has 10 classes for general documents)
    hf_repo: str = "juliozhao/DocLayout-YOLO-DocStructBench"
    hf_filename: str = "doclayout_yolo_docstructbench_imgsz1024.pt"
    device: str = "cpu"
    conf: float = 0.2
    iou: float = 0.5
    create_annotation_json: bool = True

    def __post_init__(self):
        # DocStructBench classes (general document layout)
        # See: https://github.com/opendatalab/DocLayout-YOLO
        self.class_names = {
            "0": "title",
            "1": "plain text",
            "2": "abandon",
            "3": "figure",
            "4": "figure_caption",
            "5": "table",
            "6": "table_caption",
            "7": "table_footnote",
            "8": "isolate_formula",
            "9": "formula_caption",
        }
        # Text classes for line segmentation
        self.text_classes = ["plain text", "title"]

from __future__ import annotations

from paddlex.inference.pipelines.layout_parsing.layout_objects import LayoutBlock
from paddlex.inference.pipelines.layout_parsing.result_v2 import LayoutParsingResultV2
from paddlex.inference.pipelines.ocr.result import OCRResult
from paddlex.inference.pipelines.layout_parsing.xycut_enhanced.xycuts import (
    sort_by_xycut,
)
from paddlex.inference.pipelines.layout_parsing.utils import get_sub_regions_ocr_res

__all__ = [
    "PaddleBlock",
    "PaddleResult",
]


class PaddleBlock:
    def __init__(self, label: str, bbox: list, content: str, ocr_words: list):
        self.label = label
        self.bbox = bbox
        self.content = content
        self.ocr_words = ocr_words


class PaddleResult:
    def __init__(self, result: LayoutParsingResultV2):
        self._result = result
        self.blocks = []

        overall_ocr_res = result.get("overall_ocr_res")
        parsing_res_list = result.get("parsing_res_list", [])

        if overall_ocr_res and parsing_res_list:
            for block in parsing_res_list:
                # Extract OCR words for this block
                ocr_words = self._extract_ocr_words_for_block(block, overall_ocr_res)

                # Create PaddleBlock instance
                paddle_block = PaddleBlock(
                    label=block.label,
                    bbox=block.bbox.tolist()
                    if hasattr(block.bbox, "tolist")
                    else list(block.bbox),
                    content=block.content if hasattr(block, "content") else "",
                    ocr_words=ocr_words,
                )

                self.blocks.append(paddle_block)

    @staticmethod
    def _extract_ocr_words_for_block(
        block: LayoutBlock, overall_ocr_res: OCRResult
    ) -> list:
        # Get OCR indices within this block's bbox
        _, ocr_idx_list = get_sub_regions_ocr_res(
            overall_ocr_res, [block.bbox], return_match_idx=True
        )

        # Create word dictionaries
        ocr_words = [
            {
                "bbox": overall_ocr_res["rec_boxes"][box_no].tolist()
                if hasattr(overall_ocr_res["rec_boxes"][box_no], "tolist")
                else list(overall_ocr_res["rec_boxes"][box_no]),
                "text": overall_ocr_res["rec_texts"][box_no],
                "label": overall_ocr_res["rec_labels"][box_no],
                "score": float(overall_ocr_res["rec_scores"][box_no]),
            }
            for box_no in ocr_idx_list
        ]

        # Sort words by reading order if any words exist
        if ocr_words:
            # Determine direction based on block type
            direction = (
                "vertical"
                if block.label in ["table", "seal", "chart"]
                else "horizontal"
            )
            word_bboxes = [w["bbox"] for w in ocr_words]
            sorted_indices = sort_by_xycut(word_bboxes, direction=direction, min_gap=1)
            ocr_words = [ocr_words[i] for i in sorted_indices]

        return ocr_words

    @classmethod
    def from_layout_result(cls, result: LayoutParsingResultV2) -> "PaddleResult":
        return cls(result)

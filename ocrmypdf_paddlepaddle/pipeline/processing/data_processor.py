from __future__ import annotations

import copy
from typing import Any, Union

import numpy as np
from paddlex.inference.models.object_detection.result import DetResult
from paddlex.inference.pipelines.layout_parsing.setting import BLOCK_LABEL_MAP, BLOCK_SETTINGS
from paddlex.inference.pipelines.layout_parsing.utils import (
    calculate_bbox_area,
    convert_formula_res_to_ocr_format,
    get_sub_regions_ocr_res,
    remove_overlap_blocks,
    update_region_box,
)
from paddlex.inference.pipelines.ocr.result import OCRResult

from .ocr_handler import OCRHandler
from .region_mapper import RegionMapper


class DataStandardizer:
    def __init__(self):
        self.ocr_handler = OCRHandler()
        self.region_mapper = RegionMapper()

    def standardize_data(
        self,
        image: np.ndarray,
        region_det_res: DetResult,
        layout_det_res: DetResult,
        overall_ocr_res: OCRResult,
        formula_res_list: list,
        text_rec_model: Any,
        text_rec_score_thresh: Union[float, None] = None,
        default_text_rec_score_thresh: float = 0.0,
    ) -> tuple:
        """
        Retrieves the layout parsing result based on the layout detection result, OCR result, and other recognition results.

        Args:
            image (np.ndarray): The input image.
            overall_ocr_res (OCRResult): An object containing the overall OCR results.
            layout_det_res (DetResult): An object containing the layout detection results.
            region_det_res (DetResult): Region detection results.
            formula_res_list (list): A list of formula recognition results.
            text_rec_model (Any): The text recognition model.
            text_rec_score_thresh (Optional[float], optional): The score threshold for text recognition. Defaults to None.

        Returns:
            tuple: (region_block_ocr_idx_map, region_det_res, layout_det_res)
        """
        matched_ocr_dict = {}
        block_to_ocr_map = {}
        base_region_bbox = [65535, 65535, 0, 0]
        bottom_text_y_max = 0
        max_block_area = 0.0
        doc_title_num = 0
        footnote_list = []
        paragraph_title_list = []

        layout_det_res = remove_overlap_blocks(
            layout_det_res,
            threshold=0.5,
            smaller=True,
        )

        convert_formula_res_to_ocr_format(formula_res_list, overall_ocr_res)

        for box_idx, box_info in enumerate(layout_det_res["boxes"]):
            box = box_info["coordinate"]
            label = box_info["label"].lower()
            _, _, _, y2 = box

            base_region_bbox = update_region_box(box, base_region_bbox)
            max_block_area = max(max_block_area, calculate_bbox_area(box))

            if label == "footnote":
                footnote_list.append(box_idx)
            elif label == "paragraph_title":
                paragraph_title_list.append(box_idx)
            if label == "text":
                bottom_text_y_max = max(y2, bottom_text_y_max)
            if label == "doc_title":
                doc_title_num += 1

            if label not in ["formula", "table", "seal"]:
                _, matched_idxes = get_sub_regions_ocr_res(
                    overall_ocr_res, [box], return_match_idx=True
                )
                block_to_ocr_map[box_idx] = matched_idxes
                for matched_idx in matched_idxes:
                    if matched_ocr_dict.get(matched_idx, None) is None:
                        matched_ocr_dict[matched_idx] = [box_idx]
                    else:
                        matched_ocr_dict[matched_idx].append(box_idx)

        self._fix_footnote_labels(layout_det_res, footnote_list, bottom_text_y_max)
        self._fix_paragraph_title_labels(
            layout_det_res, paragraph_title_list, doc_title_num, max_block_area
        )

        self.ocr_handler.process_hurdle_ocr(
            matched_ocr_dict,
            block_to_ocr_map,
            overall_ocr_res,
            layout_det_res,
            image,
            text_rec_model,
            text_rec_score_thresh,
            default_text_rec_score_thresh,
        )

        self.ocr_handler.ocr_empty_layout_blocks(
            block_to_ocr_map,
            overall_ocr_res,
            layout_det_res,
            image,
            text_rec_model,
            text_rec_score_thresh,
            default_text_rec_score_thresh,
        )

        self._handle_empty_layout_detection(
            layout_det_res, overall_ocr_res, block_to_ocr_map, base_region_bbox
        )

        region_to_block_map = self.region_mapper.map_regions_to_blocks(
            region_det_res,
            layout_det_res,
            base_region_bbox,
            image.shape,
        )

        region_block_ocr_idx_map = dict(
            region_to_block_map=region_to_block_map,
            block_to_ocr_map=block_to_ocr_map,
        )

        return region_block_ocr_idx_map, region_det_res, layout_det_res

    @staticmethod
    def _fix_footnote_labels(
        layout_det_res: DetResult, footnote_list: list, bottom_text_y_max: float
    ) -> None:
        """Fix the footnote label if it's above the text boxes."""
        for footnote_idx in footnote_list:
            if (
                layout_det_res["boxes"][footnote_idx]["coordinate"][3]
                < bottom_text_y_max
            ):
                layout_det_res["boxes"][footnote_idx]["label"] = "text"

    @staticmethod
    def _fix_paragraph_title_labels(
        layout_det_res: DetResult,
        paragraph_title_list: list,
        doc_title_num: int,
        max_block_area: float,
    ) -> None:
        """Check if there is only one paragraph title and without doc_title, convert it to doc_title if large enough."""
        only_one_paragraph_title = len(paragraph_title_list) == 1 and doc_title_num == 0
        if only_one_paragraph_title:
            paragraph_title_block_area = calculate_bbox_area(
                layout_det_res["boxes"][paragraph_title_list[0]]["coordinate"]
            )
            title_area_max_block_threshold = BLOCK_SETTINGS.get(
                "title_conversion_area_ratio_threshold", 0.3
            )
            if (
                paragraph_title_block_area
                > max_block_area * title_area_max_block_threshold
            ):
                layout_det_res["boxes"][paragraph_title_list[0]]["label"] = "doc_title"

    @staticmethod
    def _handle_empty_layout_detection(
        layout_det_res: DetResult,
        overall_ocr_res: OCRResult,
        block_to_ocr_map: dict,
        base_region_bbox: list,
    ) -> None:
        """When there is no layout detection result but there is OCR result, convert OCR detection result to layout detection result."""
        if len(layout_det_res["boxes"]) == 0 and len(overall_ocr_res["rec_boxes"]) > 0:
            for idx, ocr_rec_box in enumerate(overall_ocr_res["rec_boxes"]):
                base_region_bbox = update_region_box(ocr_rec_box, base_region_bbox)
                layout_det_res["boxes"].append(
                    {
                        "label": "text",
                        "coordinate": ocr_rec_box,
                        "score": overall_ocr_res["rec_scores"][idx],
                    }
                )
                block_to_ocr_map[idx] = [idx]

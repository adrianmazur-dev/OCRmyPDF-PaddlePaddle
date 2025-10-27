from __future__ import annotations

import copy
from typing import Any, Union

import numpy as np
from paddlex.inference.models.object_detection.result import DetResult
from paddlex.inference.pipelines.layout_parsing.setting import BLOCK_LABEL_MAP
from paddlex.inference.pipelines.layout_parsing.utils import (
    calculate_overlap_ratio,
    get_bbox_intersection,
    get_sub_regions_ocr_res,
)
from paddlex.inference.pipelines.ocr.result import OCRResult


class OCRHandler:
    @staticmethod
    def get_text_paragraphs_ocr_res(
        overall_ocr_res: OCRResult,
        layout_det_res: DetResult,
    ) -> OCRResult:
        """
        Retrieves the OCR results for text paragraphs, excluding those of formulas, tables, and seals.

        Args:
            overall_ocr_res (OCRResult): The overall OCR result containing text information.
            layout_det_res (DetResult): The detection result containing the layout information of the document.

        Returns:
            OCRResult: The OCR result for text paragraphs after excluding formulas, tables, and seals.
        """
        object_boxes = []
        for box_info in layout_det_res["boxes"]:
            if box_info["label"].lower() in ["formula", "table", "seal"]:
                object_boxes.append(box_info["coordinate"])
        object_boxes = np.array(object_boxes)
        sub_regions_ocr_res = get_sub_regions_ocr_res(
            overall_ocr_res, object_boxes, flag_within=False
        )
        return sub_regions_ocr_res

    @staticmethod
    def process_hurdle_ocr(
        matched_ocr_dict: dict,
        block_to_ocr_map: dict,
        overall_ocr_res: OCRResult,
        layout_det_res: DetResult,
        image: np.ndarray,
        text_rec_model: Any,
        text_rec_score_thresh: Union[float, None] = None,
        default_score_thresh: float = 0.0,
    ) -> None:
        """
        Replace the OCR information of text that crosses multiple layout blocks (hurdles).

        Args:
            matched_ocr_dict (dict): Dictionary mapping OCR indices to layout box IDs.
            block_to_ocr_map (dict): Dictionary mapping block indices to OCR indices.
            overall_ocr_res (OCRResult): The overall OCR result.
            layout_det_res (DetResult): The layout detection result.
            image (np.ndarray): The input image.
            text_rec_model (Any): The text recognition model.
            text_rec_score_thresh (Union[float, None]): The score threshold for text recognition.
        """
        for overall_ocr_idx, layout_box_ids in matched_ocr_dict.items():
            if len(layout_box_ids) > 1:
                matched_no = 0
                overall_ocr_box = copy.deepcopy(
                    overall_ocr_res["rec_boxes"][overall_ocr_idx]
                )
                overall_ocr_dt_poly = copy.deepcopy(
                    overall_ocr_res["dt_polys"][overall_ocr_idx]
                )
                for box_idx in layout_box_ids:
                    layout_box = layout_det_res["boxes"][box_idx]["coordinate"]
                    crop_box = get_bbox_intersection(overall_ocr_box, layout_box)
                    for ocr_idx in block_to_ocr_map[box_idx]:
                        ocr_box = overall_ocr_res["rec_boxes"][ocr_idx]
                        iou = calculate_overlap_ratio(ocr_box, crop_box, "small")
                        if iou > 0.8:
                            overall_ocr_res["rec_texts"][ocr_idx] = ""
                    x1, y1, x2, y2 = [int(i) for i in crop_box]
                    crop_img = np.array(image)[y1:y2, x1:x2]
                    crop_img_rec_res = list(text_rec_model([crop_img]))[0]
                    crop_img_dt_poly = get_bbox_intersection(
                        overall_ocr_dt_poly, layout_box, return_format="poly"
                    )
                    crop_img_rec_score = crop_img_rec_res["rec_score"]
                    crop_img_rec_text = crop_img_rec_res["rec_text"]

                    threshold = text_rec_score_thresh if text_rec_score_thresh is not None else default_score_thresh

                    if crop_img_rec_score >= threshold:
                        matched_no += 1
                        if matched_no == 1:
                            overall_ocr_res["dt_polys"][overall_ocr_idx] = (
                                crop_img_dt_poly
                            )
                            overall_ocr_res["rec_boxes"][overall_ocr_idx] = crop_box
                            overall_ocr_res["rec_polys"][overall_ocr_idx] = (
                                crop_img_dt_poly
                            )
                            overall_ocr_res["rec_scores"][overall_ocr_idx] = (
                                crop_img_rec_score
                            )
                            overall_ocr_res["rec_texts"][overall_ocr_idx] = (
                                crop_img_rec_text
                            )
                        else:
                            overall_ocr_res["dt_polys"].append(crop_img_dt_poly)
                            if len(overall_ocr_res["rec_boxes"]) == 0:
                                overall_ocr_res["rec_boxes"] = np.array([crop_box])
                            else:
                                overall_ocr_res["rec_boxes"] = np.vstack(
                                    (overall_ocr_res["rec_boxes"], crop_box)
                                )
                            overall_ocr_res["rec_polys"].append(crop_img_dt_poly)
                            overall_ocr_res["rec_scores"].append(crop_img_rec_score)
                            overall_ocr_res["rec_texts"].append(crop_img_rec_text)
                            overall_ocr_res["rec_labels"].append("text")
                            block_to_ocr_map[box_idx].remove(overall_ocr_idx)
                            block_to_ocr_map[box_idx].append(
                                len(overall_ocr_res["rec_texts"]) - 1
                            )

    @staticmethod
    def ocr_empty_layout_blocks(
        block_to_ocr_map: dict,
        overall_ocr_res: OCRResult,
        layout_det_res: DetResult,
        image: np.ndarray,
        text_rec_model: Any,
        text_rec_score_thresh: Union[float, None] = None,
        default_score_thresh: float = 0.0,
    ) -> None:
        """
        Use layout bbox to do OCR recognition when there is no matched OCR.

        Args:
            block_to_ocr_map (dict): Dictionary mapping block indices to OCR indices.
            overall_ocr_res (OCRResult): The overall OCR result.
            layout_det_res (DetResult): The layout detection result.
            image (np.ndarray): The input image.
            text_rec_model (Any): The text recognition model.
            text_rec_score_thresh (Union[float, None]): The score threshold for text recognition.
        """
        for layout_box_idx, overall_ocr_idxes in block_to_ocr_map.items():
            has_text = False
            for idx in overall_ocr_idxes:
                if overall_ocr_res["rec_texts"][idx] != "":
                    has_text = True
                    break
            if not has_text and layout_det_res["boxes"][layout_box_idx][
                "label"
            ] not in BLOCK_LABEL_MAP.get("vision_labels", []):
                crop_box = layout_det_res["boxes"][layout_box_idx]["coordinate"]
                x1, y1, x2, y2 = [int(i) for i in crop_box]
                crop_img = np.array(image)[y1:y2, x1:x2]
                crop_img_rec_res = list(text_rec_model([crop_img]))[0]
                crop_img_dt_poly = get_bbox_intersection(
                    crop_box, crop_box, return_format="poly"
                )
                crop_img_rec_score = crop_img_rec_res["rec_score"]
                crop_img_rec_text = crop_img_rec_res["rec_text"]

                threshold = text_rec_score_thresh if text_rec_score_thresh is not None else default_score_thresh

                if crop_img_rec_score >= threshold:
                    if len(overall_ocr_res["rec_boxes"]) == 0:
                        overall_ocr_res["rec_boxes"] = np.array([crop_box])
                    else:
                        overall_ocr_res["rec_boxes"] = np.vstack(
                            (overall_ocr_res["rec_boxes"], crop_box)
                        )
                    overall_ocr_res["rec_polys"].append(crop_img_dt_poly)
                    overall_ocr_res["rec_scores"].append(crop_img_rec_score)
                    overall_ocr_res["rec_texts"].append(crop_img_rec_text)
                    overall_ocr_res["rec_labels"].append("text")
                    block_to_ocr_map[layout_box_idx].append(
                        len(overall_ocr_res["rec_texts"]) - 1
                    )

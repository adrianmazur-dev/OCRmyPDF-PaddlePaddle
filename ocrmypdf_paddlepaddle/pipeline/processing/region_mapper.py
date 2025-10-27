from __future__ import annotations

import copy
from typing import List

import numpy as np
from paddlex.inference.models.object_detection.result import DetResult
from paddlex.inference.pipelines.layout_parsing.setting import (
    BLOCK_LABEL_MAP,
    REGION_SETTINGS,
)
from paddlex.inference.pipelines.layout_parsing.utils import (
    calculate_bbox_area,
    calculate_minimum_enclosing_bbox,
    calculate_overlap_ratio,
    shrink_supplement_region_bbox,
)


class RegionMapper:
    @staticmethod
    def map_regions_to_blocks(
        region_det_res: DetResult,
        layout_det_res: DetResult,
        base_region_bbox: list,
        image_shape: tuple,
    ) -> dict:
        """
        Map regions to layout blocks and create supplementary regions for unmatched blocks.

        Args:
            region_det_res (DetResult): Region detection results.
            layout_det_res (DetResult): Layout detection results.
            base_region_bbox (list): Base region bounding box.
            image_shape (tuple): Image shape (height, width).

        Returns:
            dict: Dictionary with region_to_block_map.
        """
        mask_labels = (
            BLOCK_LABEL_MAP.get("unordered_labels", [])
            + BLOCK_LABEL_MAP.get("header_labels", [])
            + BLOCK_LABEL_MAP.get("footer_labels", [])
        )
        block_bboxes = [box["coordinate"] for box in layout_det_res["boxes"]]
        region_det_res["boxes"] = sorted(
            region_det_res["boxes"],
            key=lambda item: calculate_bbox_area(item["coordinate"]),
        )

        region_to_block_map = {}

        if len(region_det_res["boxes"]) == 0:
            region_det_res["boxes"] = [
                {
                    "coordinate": base_region_bbox,
                    "label": "SupplementaryRegion",
                    "score": 1,
                }
            ]
            region_to_block_map[0] = range(len(block_bboxes))
        else:
            block_idxes_set = set(range(len(block_bboxes)))

            for region_idx, region_info in enumerate(region_det_res["boxes"]):
                matched_idxes = []
                region_to_block_map[region_idx] = []
                region_bbox = region_info["coordinate"]
                for block_idx in block_idxes_set:
                    if layout_det_res["boxes"][block_idx]["label"] in mask_labels:
                        continue
                    overlap_ratio = calculate_overlap_ratio(
                        region_bbox, block_bboxes[block_idx], mode="small"
                    )
                    if overlap_ratio > REGION_SETTINGS.get(
                        "match_block_overlap_ratio_threshold", 0.8
                    ):
                        matched_idxes.append(block_idx)
                old_region_bbox_matched_idxes = []
                if len(matched_idxes) > 0:
                    while len(old_region_bbox_matched_idxes) != len(matched_idxes):
                        old_region_bbox_matched_idxes = copy.deepcopy(matched_idxes)
                        matched_idxes = []
                        matched_bboxes = [
                            block_bboxes[idx] for idx in old_region_bbox_matched_idxes
                        ]
                        new_region_bbox = calculate_minimum_enclosing_bbox(
                            matched_bboxes
                        )
                        for block_idx in block_idxes_set:
                            if (
                                layout_det_res["boxes"][block_idx]["label"]
                                in mask_labels
                            ):
                                continue
                            overlap_ratio = calculate_overlap_ratio(
                                new_region_bbox, block_bboxes[block_idx], mode="small"
                            )
                            if overlap_ratio > REGION_SETTINGS.get(
                                "match_block_overlap_ratio_threshold", 0.8
                            ):
                                matched_idxes.append(block_idx)
                    for block_idx in matched_idxes:
                        block_idxes_set.remove(block_idx)
                    region_to_block_map[region_idx] = matched_idxes
                    region_det_res["boxes"][region_idx]["coordinate"] = new_region_bbox

            RegionMapper._create_supplement_regions(
                region_det_res,
                region_to_block_map,
                block_idxes_set,
                block_bboxes,
                layout_det_res,
                mask_labels,
                image_shape,
            )

        return region_to_block_map

    @staticmethod
    def _create_supplement_regions(
        region_det_res: DetResult,
        region_to_block_map: dict,
        block_idxes_set: set,
        block_bboxes: List,
        layout_det_res: DetResult,
        mask_labels: List,
        image_shape: tuple,
    ) -> None:
        """
        Create supplementary regions for blocks that haven't been matched to any region.

        Args:
            region_det_res (DetResult): Region detection results (modified in place).
            region_to_block_map (dict): Mapping of regions to blocks (modified in place).
            block_idxes_set (set): Set of unmatched block indices.
            block_bboxes (List): List of block bounding boxes.
            layout_det_res (DetResult): Layout detection results.
            mask_labels (List): Labels to mask.
            image_shape (tuple): Image shape (height, width).
        """
        while len(block_idxes_set) > 0:
            unmatched_bboxes = [block_bboxes[idx] for idx in block_idxes_set]
            if len(unmatched_bboxes) == 0:
                break
            supplement_region_bbox = calculate_minimum_enclosing_bbox(unmatched_bboxes)
            matched_idxes = []

            for region_idx, region_info in enumerate(region_det_res["boxes"]):
                if len(region_to_block_map[region_idx]) == 0:
                    continue
                region_bbox = region_info["coordinate"]
                overlap_ratio = calculate_overlap_ratio(
                    supplement_region_bbox, region_bbox
                )
                if overlap_ratio > 0:
                    supplement_region_bbox, matched_idxes = (
                        shrink_supplement_region_bbox(
                            supplement_region_bbox,
                            region_bbox,
                            image_shape[1],
                            image_shape[0],
                            block_idxes_set,
                            block_bboxes,
                        )
                    )

            matched_idxes = [
                idx
                for idx in matched_idxes
                if layout_det_res["boxes"][idx]["label"] not in mask_labels
            ]
            if len(matched_idxes) == 0:
                matched_idxes = [
                    idx
                    for idx in block_idxes_set
                    if layout_det_res["boxes"][idx]["label"] not in mask_labels
                ]
                if len(matched_idxes) == 0:
                    break
            matched_bboxes = [block_bboxes[idx] for idx in matched_idxes]
            supplement_region_bbox = calculate_minimum_enclosing_bbox(matched_bboxes)
            region_idx = len(region_det_res["boxes"])
            region_to_block_map[region_idx] = list(matched_idxes)
            for block_idx in matched_idxes:
                block_idxes_set.remove(block_idx)
            region_det_res["boxes"].append(
                {
                    "coordinate": supplement_region_bbox,
                    "label": "SupplementaryRegion",
                    "score": 1,
                }
            )

        mask_idxes = [
            idx
            for idx in range(len(layout_det_res["boxes"]))
            if layout_det_res["boxes"][idx]["label"] in mask_labels
        ]
        for idx in mask_idxes:
            bbox = layout_det_res["boxes"][idx]["coordinate"]
            region_idx = len(region_det_res["boxes"])
            region_to_block_map[region_idx] = [idx]
            region_det_res["boxes"].append(
                {
                    "coordinate": bbox,
                    "label": "SupplementaryRegion",
                    "score": 1,
                }
            )

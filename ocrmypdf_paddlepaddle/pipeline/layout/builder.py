from __future__ import annotations

from typing import Any, List, Union

import numpy as np
from paddlex.inference.models.object_detection.result import DetResult
from paddlex.inference.pipelines.layout_parsing.layout_objects import (
    LayoutBlock,
    LayoutRegion,
)
from paddlex.inference.pipelines.layout_parsing.setting import BLOCK_LABEL_MAP
from paddlex.inference.pipelines.layout_parsing.utils import (
    get_sub_regions_ocr_res,
    update_region_box,
)
from paddlex.inference.pipelines.ocr.result import OCRResult
from PIL import Image


class LayoutObjectBuilder:
    @staticmethod
    def build_layout_parsing_objects(
        image: np.ndarray,
        region_block_ocr_idx_map: dict,
        region_det_res: DetResult,
        overall_ocr_res: OCRResult,
        layout_det_res: DetResult,
        table_res_list: list,
        seal_res_list: list,
        chart_res_list: list,
        text_rec_model: Any,
        text_rec_score_thresh: Union[float, None] = None,
    ) -> LayoutRegion:
        """
        Extract structured information from OCR and layout detection results.

        Args:
            image (np.ndarray): The input image.
            region_block_ocr_idx_map (dict): Mapping of regions to blocks and OCR results.
            region_det_res (DetResult): Region detection results.
            overall_ocr_res (OCRResult): Overall OCR results.
            layout_det_res (DetResult): Layout detection results.
            table_res_list (list): Table recognition results.
            seal_res_list (list): Seal recognition results.
            chart_res_list (list): Chart recognition results.
            text_rec_model (Any): Text recognition model.
            text_rec_score_thresh (Union[float, None]): Score threshold for text recognition.

        Returns:
            LayoutRegion: The layout parsing page containing all regions.
        """
        table_index = 0
        seal_index = 0
        chart_index = 0
        layout_parsing_blocks: List[LayoutBlock] = []

        for box_idx, box_info in enumerate(layout_det_res["boxes"]):
            label = box_info["label"]
            block_bbox = box_info["coordinate"]
            rec_res = {"boxes": [], "rec_texts": [], "rec_labels": []}

            block = LayoutBlock(label=label, bbox=block_bbox)

            if label == "table" and len(table_res_list) > 0:
                block.content = table_res_list[table_index]["pred_html"]
                table_index += 1
            elif label == "seal" and len(seal_res_list) > 0:
                block.content = "\n".join(seal_res_list[seal_index]["rec_texts"])
                seal_index += 1
            elif label == "chart" and len(chart_res_list) > 0:
                block.content = chart_res_list[chart_index]
                chart_index += 1
            else:
                if label == "formula":
                    _, ocr_idx_list = get_sub_regions_ocr_res(
                        overall_ocr_res, [block_bbox], return_match_idx=True
                    )
                    region_block_ocr_idx_map["block_to_ocr_map"][box_idx] = (
                        ocr_idx_list
                    )
                else:
                    ocr_idx_list = region_block_ocr_idx_map["block_to_ocr_map"].get(
                        box_idx, []
                    )
                for box_no in ocr_idx_list:
                    rec_res["boxes"].append(overall_ocr_res["rec_boxes"][box_no])
                    rec_res["rec_texts"].append(
                        overall_ocr_res["rec_texts"][box_no],
                    )
                    rec_res["rec_labels"].append(
                        overall_ocr_res["rec_labels"][box_no],
                    )
                block.update_text_content(
                    image=image,
                    ocr_rec_res=rec_res,
                    text_rec_model=text_rec_model,
                    text_rec_score_thresh=text_rec_score_thresh,
                )

            if (
                label
                in ["seal", "table", "formula", "chart"]
                + BLOCK_LABEL_MAP["image_labels"]
            ):
                x_min, y_min, x_max, y_max = list(map(int, block_bbox))
                img_path = (
                    f"imgs/img_in_{block.label}_box_{x_min}_{y_min}_{x_max}_{y_max}.jpg"
                )
                img = Image.fromarray(image[y_min:y_max, x_min:x_max, ::-1])
                block.image = {"path": img_path, "img": img}

            layout_parsing_blocks.append(block)

        page_region_bbox = [65535, 65535, 0, 0]
        layout_parsing_regions: List[LayoutRegion] = []
        for region_idx, region_info in enumerate(region_det_res["boxes"]):
            region_bbox = np.array(region_info["coordinate"]).astype("int")
            region_blocks = [
                layout_parsing_blocks[idx]
                for idx in region_block_ocr_idx_map["region_to_block_map"][region_idx]
            ]
            if region_blocks:
                page_region_bbox = update_region_box(region_bbox, page_region_bbox)
                region = LayoutRegion(bbox=region_bbox, blocks=region_blocks)
                layout_parsing_regions.append(region)

        layout_parsing_page = LayoutRegion(
            bbox=np.array(page_region_bbox).astype("int"), blocks=layout_parsing_regions
        )

        return layout_parsing_page

from __future__ import annotations

import re
from typing import Any, List, Union

import numpy as np
from paddlex.inference.models.object_detection.result import DetResult
from paddlex.inference.pipelines.layout_parsing.layout_objects import LayoutRegion
from paddlex.inference.pipelines.layout_parsing.setting import BLOCK_LABEL_MAP
from paddlex.inference.pipelines.ocr.result import OCRResult

from ..layout.builder import LayoutObjectBuilder
from ..layout.sorter import BlockSorter
from ..processing.data_processor import DataStandardizer


class ResultAggregator:
    def __init__(self, data_standardizer: DataStandardizer):
        self.data_standardizer = data_standardizer
        self.layout_builder = LayoutObjectBuilder()
        self.block_sorter = BlockSorter()

    def get_layout_parsing_res(
        self,
        image: np.ndarray,
        region_det_res: DetResult,
        layout_det_res: DetResult,
        overall_ocr_res: OCRResult,
        table_res_list: list,
        seal_res_list: list,
        chart_res_list: list,
        formula_res_list: list,
        text_rec_model: Any,
        text_rec_score_thresh: Union[float, None] = None,
        default_text_rec_score_thresh: float = 0.0,
    ) -> list:
        """
        Retrieves the layout parsing result based on the layout detection result, OCR result, and other recognition results.

        Args:
            image (np.ndarray): The input image.
            layout_det_res (DetResult): The detection result containing the layout information of the document.
            overall_ocr_res (OCRResult): The overall OCR result containing text information.
            table_res_list (list): A list of table recognition results.
            seal_res_list (list): A list of seal recognition results.
            chart_res_list (list): A list of chart recognition results.
            formula_res_list (list): A list of formula recognition results.
            text_rec_score_thresh (Optional[float], optional): The score threshold for text recognition. Defaults to None.

        Returns:
            list: A list of dictionaries representing the layout parsing result.
        """
        region_block_ocr_idx_map, region_det_res, layout_det_res = (
            self.data_standardizer.standardize_data(
                image=image,
                region_det_res=region_det_res,
                layout_det_res=layout_det_res,
                overall_ocr_res=overall_ocr_res,
                formula_res_list=formula_res_list,
                text_rec_model=text_rec_model,
                text_rec_score_thresh=text_rec_score_thresh,
                default_text_rec_score_thresh=default_text_rec_score_thresh,
            )
        )

        layout_parsing_page = self.layout_builder.build_layout_parsing_objects(
            image=image,
            region_block_ocr_idx_map=region_block_ocr_idx_map,
            region_det_res=region_det_res,
            overall_ocr_res=overall_ocr_res,
            layout_det_res=layout_det_res,
            table_res_list=table_res_list,
            seal_res_list=seal_res_list,
            chart_res_list=chart_res_list,
            text_rec_model=text_rec_model,
            text_rec_score_thresh=text_rec_score_thresh,
        )

        parsing_res_list = self.block_sorter.sort_layout_parsing_blocks(
            layout_parsing_page
        )

        order_index = 1
        for index, block in enumerate(parsing_res_list):
            block.index = index
            if block.label in BLOCK_LABEL_MAP["visualize_index_labels"]:
                block.order_index = order_index
                order_index += 1

        return parsing_res_list

    @staticmethod
    def concatenate_markdown_pages(markdown_list: list) -> str:
        """
        Concatenate Markdown content from multiple pages into a single document.

        Args:
            markdown_list (list): A list containing Markdown data for each page.

        Returns:
            str: The processed Markdown text.
        """
        markdown_texts = ""
        previous_page_last_element_paragraph_end_flag = True

        for res in markdown_list:
            page_first_element_paragraph_start_flag: bool = res[
                "page_continuation_flags"
            ][0]
            page_last_element_paragraph_end_flag: bool = res["page_continuation_flags"][
                1
            ]

            if (
                not page_first_element_paragraph_start_flag
                and not previous_page_last_element_paragraph_end_flag
            ):
                last_char_of_markdown = markdown_texts[-1] if markdown_texts else ""
                first_char_of_handler = (
                    res["markdown_texts"][0] if res["markdown_texts"] else ""
                )

                last_is_chinese_char = (
                    re.match(r"[\u4e00-\u9fff]", last_char_of_markdown)
                    if last_char_of_markdown
                    else False
                )
                first_is_chinese_char = (
                    re.match(r"[\u4e00-\u9fff]", first_char_of_handler)
                    if first_char_of_handler
                    else False
                )
                if not (last_is_chinese_char or first_is_chinese_char):
                    markdown_texts += " " + res["markdown_texts"]
                else:
                    markdown_texts += res["markdown_texts"]
            else:
                markdown_texts += "\n\n" + res["markdown_texts"]
            previous_page_last_element_paragraph_end_flag = (
                page_last_element_paragraph_end_flag
            )

        return markdown_texts

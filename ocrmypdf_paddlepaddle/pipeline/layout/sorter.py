from __future__ import annotations

from typing import List

from paddlex.inference.pipelines.layout_parsing.layout_objects import (
    LayoutBlock,
    LayoutRegion,
)
from paddlex.inference.pipelines.layout_parsing.xycut_enhanced import xycut_enhanced


class BlockSorter:
    @staticmethod
    def sort_layout_parsing_blocks(layout_parsing_page: LayoutRegion) -> List[LayoutBlock]:
        """
        Sort layout parsing blocks using XYCut algorithm.

        Args:
            layout_parsing_page (LayoutRegion): The layout parsing page containing regions.

        Returns:
            List[LayoutBlock]: Sorted list of layout blocks.
        """
        layout_parsing_regions = xycut_enhanced(layout_parsing_page)
        parsing_res_list = []
        for region in layout_parsing_regions:
            layout_parsing_blocks = xycut_enhanced(region)
            parsing_res_list.extend(layout_parsing_blocks)

        return parsing_res_list

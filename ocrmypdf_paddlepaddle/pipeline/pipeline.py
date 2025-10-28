from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from paddlex.inference.common.batch_sampler import ImageBatchSampler
from paddlex.inference.common.reader import ReadImage
from paddlex.inference.pipelines._parallel import (
    AutoParallelImageSimpleInferencePipeline,
)
from paddlex.inference.pipelines.base import BasePipeline
from paddlex.inference.pipelines.layout_parsing.result_v2 import LayoutParsingResultV2
from paddlex.inference.pipelines.layout_parsing.utils import (
    convert_formula_res_to_ocr_format,
    gather_imgs,
    get_bbox_intersection,
)
from paddlex.inference.utils.benchmark import benchmark
from paddlex.inference.utils.hpi import HPIConfig
from paddlex.inference.utils.pp_option import PaddlePredictorOption
from paddlex.utils.deps import pipeline_requires_extra

from .config import ModelSettings, validate_model_settings
from .initialization import PredictorInitializer
from .processing import DataStandardizer
from .results import ResultAggregator


@benchmark.time_methods
class _LayoutParsingPipelineCustom(BasePipeline):
    def __init__(
        self,
        config: dict,
        device: str = None,
        pp_option: PaddlePredictorOption = None,
        use_hpip: bool = False,
        hpi_config: Optional[Union[Dict[str, Any], HPIConfig]] = None,
    ) -> None:
        """Initializes the layout parsing pipeline.

        Args:
            config (Dict): Configuration dictionary containing various settings.
            device (str, optional): Device to run the predictions on. Defaults to None.
            pp_option (PaddlePredictorOption, optional): PaddlePredictor options. Defaults to None.
            use_hpip (bool, optional): Whether to use the high-performance
                inference plugin (HPIP) by default. Defaults to False.
            hpi_config (Optional[Union[Dict[str, Any], HPIConfig]], optional):
                The default high-performance inference configuration dictionary.
                Defaults to None.
        """

        super().__init__(
            device=device,
            pp_option=pp_option,
            use_hpip=use_hpip,
            hpi_config=hpi_config,
        )

        self._initialize_predictors(config)
        self._initialize_processors()

        self.batch_sampler = ImageBatchSampler(batch_size=config.get("batch_size", 1))
        self.img_reader = ReadImage(format="BGR")

    def _initialize_predictors(self, config: dict) -> None:
        """Initialize all predictors and models."""
        initializer = PredictorInitializer(self)
        init_result = initializer.initialize_predictors(config)

        predictors = init_result["predictors"]
        settings = init_result["settings"]

        for name, predictor in predictors.items():
            setattr(self, name, predictor)

        self.use_doc_preprocessor = settings["use_doc_preprocessor"]
        self.use_table_recognition = settings["use_table_recognition"]
        self.use_seal_recognition = settings["use_seal_recognition"]
        self.format_block_content = settings["format_block_content"]
        self.use_region_detection = settings["use_region_detection"]
        self.use_formula_recognition = settings["use_formula_recognition"]
        self.use_chart_recognition = settings["use_chart_recognition"]

        self._default_settings = settings

    def _initialize_processors(self) -> None:
        """Initialize processing components."""
        self.data_standardizer = DataStandardizer()
        self.result_aggregator = ResultAggregator(self.data_standardizer)

    def close(self):
        if hasattr(self, "chart_recognition_model"):
            self.chart_recognition_model.close()

    def predict(
        self,
        input: Union[str, list[str], np.ndarray, list[np.ndarray]],
        use_doc_orientation_classify: Union[bool, None] = None,
        use_doc_unwarping: Union[bool, None] = None,
        use_textline_orientation: Optional[bool] = None,
        use_seal_recognition: Union[bool, None] = None,
        use_table_recognition: Union[bool, None] = None,
        use_formula_recognition: Union[bool, None] = None,
        use_chart_recognition: Union[bool, None] = None,
        use_region_detection: Union[bool, None] = None,
        format_block_content: Union[bool, None] = None,
        layout_threshold: Optional[Union[float, dict]] = None,
        layout_nms: Optional[bool] = None,
        layout_unclip_ratio: Optional[Union[float, Tuple[float, float], dict]] = None,
        layout_merge_bboxes_mode: Optional[str] = None,
        text_det_limit_side_len: Union[int, None] = None,
        text_det_limit_type: Union[str, None] = None,
        text_det_thresh: Union[float, None] = None,
        text_det_box_thresh: Union[float, None] = None,
        text_det_unclip_ratio: Union[float, None] = None,
        text_rec_score_thresh: Union[float, None] = None,
        seal_det_limit_side_len: Union[int, None] = None,
        seal_det_limit_type: Union[str, None] = None,
        seal_det_thresh: Union[float, None] = None,
        seal_det_box_thresh: Union[float, None] = None,
        seal_det_unclip_ratio: Union[float, None] = None,
        seal_rec_score_thresh: Union[float, None] = None,
        use_wired_table_cells_trans_to_html: bool = False,
        use_wireless_table_cells_trans_to_html: bool = False,
        use_table_orientation_classify: bool = True,
        use_ocr_results_with_table_cells: bool = True,
        use_e2e_wired_table_rec_model: bool = False,
        use_e2e_wireless_table_rec_model: bool = True,
        **kwargs,
    ) -> LayoutParsingResultV2:
        """
        Predicts the layout parsing result for the given input.

        Args:
            input (Union[str, list[str], np.ndarray, list[np.ndarray]]): Input image path, list of image paths,
                                                                        numpy array of an image, or list of numpy arrays.
            use_doc_orientation_classify (Optional[bool]): Whether to use document orientation classification.
            use_doc_unwarping (Optional[bool]): Whether to use document unwarping.
            use_textline_orientation (Optional[bool]): Whether to use textline orientation prediction.
            use_seal_recognition (Optional[bool]): Whether to use seal recognition.
            use_table_recognition (Optional[bool]): Whether to use table recognition.
            use_formula_recognition (Optional[bool]): Whether to use formula recognition.
            use_region_detection (Optional[bool]): Whether to use region detection.
            format_block_content (Optional[bool]): Whether to format block content.
            [... other parameters ...]
            **kwargs (Any): Additional settings to extend functionality.

        Returns:
            LayoutParsingResultV2: The predicted layout parsing result.
        """
        model_settings = ModelSettings.from_params(
            self._default_settings,
            use_doc_orientation_classify,
            use_doc_unwarping,
            use_seal_recognition,
            use_table_recognition,
            use_formula_recognition,
            use_chart_recognition,
            use_region_detection,
            format_block_content,
        )

        if not validate_model_settings(
            model_settings.to_dict(), self._default_settings
        ):
            yield {"error": "the input params for model settings are invalid!"}

        for batch_data in self.batch_sampler(input):
            image_arrays = self.img_reader(batch_data.instances)

            doc_preprocessor_images = self._run_doc_preprocessing(
                image_arrays,
                model_settings,
                use_doc_orientation_classify,
                use_doc_unwarping,
            )

            layout_det_results = self._run_layout_detection(
                doc_preprocessor_images,
                layout_threshold,
                layout_nms,
                layout_unclip_ratio,
                layout_merge_bboxes_mode,
            )

            imgs_in_doc = [
                gather_imgs(img, res["boxes"])
                for img, res in zip(doc_preprocessor_images, layout_det_results)
            ]

            region_det_results = self._run_region_detection(
                doc_preprocessor_images, model_settings
            )

            formula_res_lists = self._run_formula_recognition(
                doc_preprocessor_images,
                layout_det_results,
                model_settings,
                use_doc_orientation_classify,
                use_doc_unwarping,
            )

            self._mask_formula_regions(doc_preprocessor_images, formula_res_lists)

            overall_ocr_results = self._run_ocr(
                doc_preprocessor_images,
                use_textline_orientation,
                text_det_limit_side_len,
                text_det_limit_type,
                text_det_thresh,
                text_det_box_thresh,
                text_det_unclip_ratio,
                text_rec_score_thresh,
            )

            table_res_lists = self._run_table_recognition(
                doc_preprocessor_images,
                layout_det_results,
                overall_ocr_results,
                formula_res_lists,
                imgs_in_doc,
                model_settings,
                use_wired_table_cells_trans_to_html,
                use_wireless_table_cells_trans_to_html,
                use_table_orientation_classify,
                use_ocr_results_with_table_cells,
                use_e2e_wired_table_rec_model,
                use_e2e_wireless_table_rec_model,
            )

            seal_res_lists = self._run_seal_recognition(
                doc_preprocessor_images,
                layout_det_results,
                model_settings,
                seal_det_limit_side_len,
                seal_det_limit_type,
                seal_det_thresh,
                seal_det_box_thresh,
                seal_det_unclip_ratio,
                seal_rec_score_thresh,
            )

            for (
                input_path,
                page_index,
                doc_preprocessor_image,
                doc_preprocessor_res,
                layout_det_res,
                region_det_res,
                overall_ocr_res,
                table_res_list,
                seal_res_list,
                formula_res_list,
                imgs_in_doc_for_img,
            ) in zip(
                batch_data.input_paths,
                batch_data.page_indexes,
                doc_preprocessor_images,
                [{"output_img": arr} for arr in image_arrays],
                layout_det_results,
                region_det_results,
                overall_ocr_results,
                table_res_lists,
                seal_res_lists,
                formula_res_lists,
                imgs_in_doc,
            ):
                chart_res_list = self._run_chart_recognition(
                    doc_preprocessor_image, layout_det_res, model_settings
                )

                parsing_res_list = self.result_aggregator.get_layout_parsing_res(
                    doc_preprocessor_image,
                    region_det_res=region_det_res,
                    layout_det_res=layout_det_res,
                    overall_ocr_res=overall_ocr_res,
                    table_res_list=table_res_list,
                    seal_res_list=seal_res_list,
                    chart_res_list=chart_res_list,
                    formula_res_list=formula_res_list,
                    text_rec_model=self.general_ocr_pipeline.text_rec_model,
                    text_rec_score_thresh=text_rec_score_thresh,
                    default_text_rec_score_thresh=self.general_ocr_pipeline.text_rec_score_thresh,
                )

                self._restore_formula_regions(doc_preprocessor_image, formula_res_list)

                single_img_res = {
                    "input_path": input_path,
                    "page_index": page_index,
                    "doc_preprocessor_res": doc_preprocessor_res,
                    "layout_det_res": layout_det_res,
                    "region_det_res": region_det_res,
                    "overall_ocr_res": overall_ocr_res,
                    "table_res_list": table_res_list,
                    "seal_res_list": seal_res_list,
                    "chart_res_list": chart_res_list,
                    "formula_res_list": formula_res_list,
                    "parsing_res_list": parsing_res_list,
                    "imgs_in_doc": imgs_in_doc_for_img,
                    "model_settings": model_settings.to_dict(),
                }
                yield LayoutParsingResultV2(single_img_res)

    def _run_doc_preprocessing(
        self,
        image_arrays: list,
        model_settings: ModelSettings,
        use_doc_orientation_classify: Union[bool, None],
        use_doc_unwarping: Union[bool, None],
    ) -> list:
        """Run document preprocessing if enabled."""
        if model_settings.use_doc_preprocessor:
            doc_preprocessor_results = list(
                self.doc_preprocessor_pipeline(
                    image_arrays,
                    use_doc_orientation_classify=use_doc_orientation_classify,
                    use_doc_unwarping=use_doc_unwarping,
                )
            )
            return [item["output_img"] for item in doc_preprocessor_results]
        return image_arrays

    def _run_layout_detection(
        self,
        images: list,
        threshold: Optional[Union[float, dict]],
        layout_nms: Optional[bool],
        layout_unclip_ratio: Optional[Union[float, Tuple[float, float], dict]],
        layout_merge_bboxes_mode: Optional[str],
    ) -> list:
        """Run layout detection on images."""
        return list(
            self.layout_det_model(
                images,
                threshold=threshold,
                layout_nms=layout_nms,
                layout_unclip_ratio=layout_unclip_ratio,
                layout_merge_bboxes_mode=layout_merge_bboxes_mode,
            )
        )

    def _run_region_detection(
        self, images: list, model_settings: ModelSettings
    ) -> list:
        """Run region detection if enabled."""
        if model_settings.use_region_detection:
            return list(
                self.region_detection_model(
                    images,
                    layout_nms=True,
                    layout_merge_bboxes_mode="small",
                ),
            )
        return [{"boxes": []} for _ in images]

    def _run_formula_recognition(
        self,
        images: list,
        layout_det_results: list,
        model_settings: ModelSettings,
        use_doc_orientation_classify: Union[bool, None],
        use_doc_unwarping: Union[bool, None],
    ) -> list:
        """Run formula recognition if enabled."""
        if model_settings.use_formula_recognition:
            formula_res_all = list(
                self.formula_recognition_pipeline(
                    images,
                    use_layout_detection=False,
                    use_doc_orientation_classify=use_doc_orientation_classify,
                    use_doc_unwarping=use_doc_unwarping,
                    layout_det_res=layout_det_results,
                ),
            )
            return [item["formula_res_list"] for item in formula_res_all]
        return [[] for _ in images]

    def _mask_formula_regions(self, images: list, formula_res_lists: list) -> None:
        """Mask formula regions in images."""
        for image, formula_res_list in zip(images, formula_res_lists):
            for formula_res in formula_res_list:
                x_min, y_min, x_max, y_max = list(map(int, formula_res["dt_polys"]))
                image[y_min:y_max, x_min:x_max, :] = 255.0

    def _restore_formula_regions(
        self, image: np.ndarray, formula_res_list: list
    ) -> None:
        """Restore formula regions in image."""
        for formula_res in formula_res_list:
            x_min, y_min, x_max, y_max = list(map(int, formula_res["dt_polys"]))
            image[y_min:y_max, x_min:x_max, :] = formula_res["input_img"]

    def _run_ocr(
        self,
        images: list,
        use_textline_orientation: Optional[bool],
        text_det_limit_side_len: Union[int, None],
        text_det_limit_type: Union[str, None],
        text_det_thresh: Union[float, None],
        text_det_box_thresh: Union[float, None],
        text_det_unclip_ratio: Union[float, None],
        text_rec_score_thresh: Union[float, None],
    ) -> list:
        """Run OCR on images."""
        overall_ocr_results = list(
            self.general_ocr_pipeline(
                images,
                use_textline_orientation=use_textline_orientation,
                text_det_limit_side_len=text_det_limit_side_len,
                text_det_limit_type=text_det_limit_type,
                text_det_thresh=text_det_thresh,
                text_det_box_thresh=text_det_box_thresh,
                text_det_unclip_ratio=text_det_unclip_ratio,
                text_rec_score_thresh=text_rec_score_thresh,
            ),
        )

        for overall_ocr_res in overall_ocr_results:
            overall_ocr_res["rec_labels"] = ["text"] * len(overall_ocr_res["rec_texts"])

        return overall_ocr_results

    def _run_table_recognition(
        self,
        images: list,
        layout_det_results: list,
        overall_ocr_results: list,
        formula_res_lists: list,
        imgs_in_doc: list,
        model_settings: ModelSettings,
        use_wired_table_cells_trans_to_html: bool,
        use_wireless_table_cells_trans_to_html: bool,
        use_table_orientation_classify: bool,
        use_ocr_results_with_table_cells: bool,
        use_e2e_wired_table_rec_model: bool,
        use_e2e_wireless_table_rec_model: bool,
    ) -> list:
        """Run table recognition if enabled."""
        if not model_settings.use_table_recognition:
            return [[] for _ in images]

        table_res_lists = []
        for (
            layout_det_res,
            image,
            overall_ocr_res,
            formula_res_list,
            imgs_in_doc_for_img,
        ) in zip(
            layout_det_results,
            images,
            overall_ocr_results,
            formula_res_lists,
            imgs_in_doc,
        ):
            table_contents_for_img = self._prepare_table_contents(
                overall_ocr_res, formula_res_list, imgs_in_doc_for_img
            )

            table_res_all = list(
                self.table_recognition_pipeline(
                    image,
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_layout_detection=False,
                    use_ocr_model=False,
                    overall_ocr_res=table_contents_for_img,
                    layout_det_res=layout_det_res,
                    cell_sort_by_y_projection=True,
                    use_wired_table_cells_trans_to_html=use_wired_table_cells_trans_to_html,
                    use_wireless_table_cells_trans_to_html=use_wireless_table_cells_trans_to_html,
                    use_table_orientation_classify=use_table_orientation_classify,
                    use_ocr_results_with_table_cells=use_ocr_results_with_table_cells,
                    use_e2e_wired_table_rec_model=use_e2e_wired_table_rec_model,
                    use_e2e_wireless_table_rec_model=use_e2e_wireless_table_rec_model,
                ),
            )
            single_table_res_lists = [item["table_res_list"] for item in table_res_all]
            table_res_lists.extend(single_table_res_lists)

        return table_res_lists

    def _prepare_table_contents(
        self, overall_ocr_res: dict, formula_res_list: list, imgs_in_doc_for_img: list
    ) -> dict:
        """Prepare table contents by adding formula and image information."""
        table_contents = copy.deepcopy(overall_ocr_res)

        for formula_res in formula_res_list:
            x_min, y_min, x_max, y_max = list(map(int, formula_res["dt_polys"]))
            poly_points = [
                (x_min, y_min),
                (x_max, y_min),
                (x_max, y_max),
                (x_min, y_max),
            ]
            table_contents["dt_polys"].append(poly_points)
            rec_formula = formula_res["rec_formula"]
            if not rec_formula.startswith("$") or not rec_formula.endswith("$"):
                rec_formula = f"${rec_formula}$"
            table_contents["rec_texts"].append(f"{rec_formula}")
            if table_contents["rec_boxes"].size == 0:
                table_contents["rec_boxes"] = np.array([formula_res["dt_polys"]])
            else:
                table_contents["rec_boxes"] = np.vstack(
                    (table_contents["rec_boxes"], [formula_res["dt_polys"]])
                )
            table_contents["rec_polys"].append(poly_points)
            table_contents["rec_scores"].append(1)

        for img in imgs_in_doc_for_img:
            img_path = img["path"]
            x_min, y_min, x_max, y_max = img["coordinate"]
            poly_points = [
                (x_min, y_min),
                (x_max, y_min),
                (x_max, y_max),
                (x_min, y_max),
            ]
            table_contents["dt_polys"].append(poly_points)
            table_contents["rec_texts"].append(
                f'<div style="text-align: center;"><img src="{img_path}" alt="Image" /></div>'
            )
            if table_contents["rec_boxes"].size == 0:
                table_contents["rec_boxes"] = np.array([img["coordinate"]])
            else:
                table_contents["rec_boxes"] = np.vstack(
                    (table_contents["rec_boxes"], img["coordinate"])
                )
            table_contents["rec_polys"].append(poly_points)
            table_contents["rec_scores"].append(img["score"])

        return table_contents

    def _run_seal_recognition(
        self,
        images: list,
        layout_det_results: list,
        model_settings: ModelSettings,
        seal_det_limit_side_len: Union[int, None],
        seal_det_limit_type: Union[str, None],
        seal_det_thresh: Union[float, None],
        seal_det_box_thresh: Union[float, None],
        seal_det_unclip_ratio: Union[float, None],
        seal_rec_score_thresh: Union[float, None],
    ) -> list:
        """Run seal recognition if enabled."""
        if model_settings.use_seal_recognition:
            seal_res_all = list(
                self.seal_recognition_pipeline(
                    images,
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_layout_detection=False,
                    layout_det_res=layout_det_results,
                    seal_det_limit_side_len=seal_det_limit_side_len,
                    seal_det_limit_type=seal_det_limit_type,
                    seal_det_thresh=seal_det_thresh,
                    seal_det_box_thresh=seal_det_box_thresh,
                    seal_det_unclip_ratio=seal_det_unclip_ratio,
                    seal_rec_score_thresh=seal_rec_score_thresh,
                ),
            )
            return [item["seal_res_list"] for item in seal_res_all]
        return [[] for _ in images]

    def _run_chart_recognition(
        self, image: np.ndarray, layout_det_res: dict, model_settings: ModelSettings
    ) -> list:
        """Run chart recognition if enabled."""
        chart_res_list = []
        if model_settings.use_chart_recognition:
            chart_imgs_list = []
            for bbox in layout_det_res["boxes"]:
                if bbox["label"] == "chart":
                    x_min, y_min, x_max, y_max = bbox["coordinate"]
                    chart_img = image[
                        int(y_min) : int(y_max), int(x_min) : int(x_max), :
                    ]
                    chart_imgs_list.append({"image": chart_img})

            for chart_res_batch in self.chart_recognition_model(input=chart_imgs_list):
                chart_res_list.append(chart_res_batch["result"])

        return chart_res_list

    @staticmethod
    def concatenate_markdown_pages(markdown_list: list) -> str:
        """
        Concatenate Markdown content from multiple pages into a single document.

        Args:
            markdown_list (list): A list containing Markdown data for each page.

        Returns:
            str: The processed Markdown text.
        """
        return ResultAggregator.concatenate_markdown_pages(markdown_list)


@pipeline_requires_extra("ocr")
class LayoutParsingPipelineCustom(AutoParallelImageSimpleInferencePipeline):
    entities = ["PP-StructureCustom"]

    @property
    def _pipeline_cls(self):
        return _LayoutParsingPipelineCustom

    def _get_batch_size(self, config):
        return config.get("batch_size", 1)

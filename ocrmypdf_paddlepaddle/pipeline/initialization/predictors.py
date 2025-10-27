from __future__ import annotations

from typing import Any


class PredictorInitializer:
    def __init__(self, base_pipeline):
        self.pipeline = base_pipeline

    def initialize_predictors(self, config: dict) -> dict:
        """Initializes the predictor based on the provided configuration.

        Args:
            config (Dict): A dictionary containing the configuration for the predictor.

        Returns:
            dict: Dictionary containing initialized predictors and settings.
        """

        predictors = {}
        settings = {}

        if (
            config.get("use_doc_preprocessor", True)
            or config.get("use_doc_orientation_classify", True)
            or config.get("use_doc_unwarping", True)
        ):
            settings["use_doc_preprocessor"] = True
        else:
            settings["use_doc_preprocessor"] = False

        settings["use_doc_preprocessor"] = False

        settings["use_table_recognition"] = config.get("use_table_recognition", True)
        settings["use_seal_recognition"] = config.get("use_seal_recognition", True)
        settings["format_block_content"] = config.get("format_block_content", False)
        settings["use_region_detection"] = config.get("use_region_detection", True)
        settings["use_formula_recognition"] = config.get(
            "use_formula_recognition", True
        )
        settings["use_chart_recognition"] = config.get("use_chart_recognition", False)

        if settings["use_doc_preprocessor"]:
            doc_preprocessor_config = config.get("SubPipelines", {}).get(
                "DocPreprocessor",
                {
                    "pipeline_config_error": "config error for doc_preprocessor_pipeline!",
                },
            )
            predictors["doc_preprocessor_pipeline"] = self.pipeline.create_pipeline(
                doc_preprocessor_config,
            )

        if settings["use_region_detection"]:
            region_detection_config = config.get("SubModules", {}).get(
                "RegionDetection",
                {
                    "model_config_error": "config error for block_region_detection_model!"
                },
            )
            predictors["region_detection_model"] = self.pipeline.create_model(
                region_detection_config,
            )

        layout_det_config = config.get("SubModules", {}).get(
            "LayoutDetection",
            {"model_config_error": "config error for layout_det_model!"},
        )
        layout_kwargs = {}
        if (threshold := layout_det_config.get("threshold", None)) is not None:
            layout_kwargs["threshold"] = threshold
        if (layout_nms := layout_det_config.get("layout_nms", None)) is not None:
            layout_kwargs["layout_nms"] = layout_nms
        if (
            layout_unclip_ratio := layout_det_config.get("layout_unclip_ratio", None)
        ) is not None:
            layout_kwargs["layout_unclip_ratio"] = layout_unclip_ratio
        if (
            layout_merge_bboxes_mode := layout_det_config.get(
                "layout_merge_bboxes_mode", None
            )
        ) is not None:
            layout_kwargs["layout_merge_bboxes_mode"] = layout_merge_bboxes_mode
        predictors["layout_det_model"] = self.pipeline.create_model(
            layout_det_config, **layout_kwargs
        )

        general_ocr_config = config.get("SubPipelines", {}).get(
            "GeneralOCR",
            {"pipeline_config_error": "config error for general_ocr_pipeline!"},
        )
        predictors["general_ocr_pipeline"] = self.pipeline.create_pipeline(
            general_ocr_config,
        )

        if settings["use_seal_recognition"]:
            seal_recognition_config = config.get("SubPipelines", {}).get(
                "SealRecognition",
                {
                    "pipeline_config_error": "config error for seal_recognition_pipeline!",
                },
            )
            predictors["seal_recognition_pipeline"] = self.pipeline.create_pipeline(
                seal_recognition_config,
            )

        if settings["use_table_recognition"]:
            table_recognition_config = config.get("SubPipelines", {}).get(
                "TableRecognition",
                {
                    "pipeline_config_error": "config error for table_recognition_pipeline!",
                },
            )
            predictors["table_recognition_pipeline"] = self.pipeline.create_pipeline(
                table_recognition_config,
            )

        if settings["use_formula_recognition"]:
            formula_recognition_config = config.get("SubPipelines", {}).get(
                "FormulaRecognition",
                {
                    "pipeline_config_error": "config error for formula_recognition_pipeline!",
                },
            )
            predictors["formula_recognition_pipeline"] = self.pipeline.create_pipeline(
                formula_recognition_config,
            )

        if settings["use_chart_recognition"]:
            chart_recognition_config = config.get("SubModules", {}).get(
                "ChartRecognition",
                {
                    "model_config_error": "config error for block_region_detection_model!"
                },
            )
            predictors["chart_recognition_model"] = self.pipeline.create_model(
                chart_recognition_config,
            )

        return {"predictors": predictors, "settings": settings}

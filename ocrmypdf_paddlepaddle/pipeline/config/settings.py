from __future__ import annotations

from typing import Optional, Union
from paddlex.utils import logging


class ModelSettings:
    def __init__(
        self,
        use_doc_preprocessor: bool,
        use_seal_recognition: bool,
        use_table_recognition: bool,
        use_formula_recognition: bool,
        use_chart_recognition: bool,
        use_region_detection: bool,
        format_block_content: bool,
    ):
        self.use_doc_preprocessor = use_doc_preprocessor
        self.use_seal_recognition = use_seal_recognition
        self.use_table_recognition = use_table_recognition
        self.use_formula_recognition = use_formula_recognition
        self.use_chart_recognition = use_chart_recognition
        self.use_region_detection = use_region_detection
        self.format_block_content = format_block_content

    @classmethod
    def from_params(
        cls,
        default_settings: dict,
        use_doc_orientation_classify: Union[bool, None],
        use_doc_unwarping: Union[bool, None],
        use_seal_recognition: Union[bool, None],
        use_table_recognition: Union[bool, None],
        use_formula_recognition: Union[bool, None],
        use_chart_recognition: Union[bool, None],
        use_region_detection: Union[bool, None],
        format_block_content: Union[bool, None],
    ) -> "ModelSettings":
        """
        Get the model settings based on the provided parameters or default values.

        Args:
            default_settings (dict): Default settings from initialization.
            use_doc_orientation_classify (Union[bool, None]): Enables document orientation classification if True. Defaults to system setting if None.
            use_doc_unwarping (Union[bool, None]): Enables document unwarping if True. Defaults to system setting if None.
            use_seal_recognition (Union[bool, None]): Enables seal recognition if True. Defaults to system setting if None.
            use_table_recognition (Union[bool, None]): Enables table recognition if True. Defaults to system setting if None.
            use_formula_recognition (Union[bool, None]): Enables formula recognition if True. Defaults to system setting if None.
            format_block_content (Union[bool, None]): Enables block content formatting if True. Defaults to system setting if None.

        Returns:
            ModelSettings: A ModelSettings instance containing the model settings.

        """
        if use_doc_orientation_classify is None and use_doc_unwarping is None:
            use_doc_preprocessor = default_settings["use_doc_preprocessor"]
        else:
            if use_doc_orientation_classify is True or use_doc_unwarping is True:
                use_doc_preprocessor = True
            else:
                use_doc_preprocessor = False

        if use_seal_recognition is None:
            use_seal_recognition = default_settings["use_seal_recognition"]

        if use_table_recognition is None:
            use_table_recognition = default_settings["use_table_recognition"]

        if use_formula_recognition is None:
            use_formula_recognition = default_settings["use_formula_recognition"]

        if use_region_detection is None:
            use_region_detection = default_settings["use_region_detection"]

        if use_chart_recognition is None:
            use_chart_recognition = default_settings["use_chart_recognition"]

        if format_block_content is None:
            format_block_content = default_settings["format_block_content"]

        return cls(
            use_doc_preprocessor=use_doc_preprocessor,
            use_seal_recognition=use_seal_recognition,
            use_table_recognition=use_table_recognition,
            use_formula_recognition=use_formula_recognition,
            use_chart_recognition=use_chart_recognition,
            use_region_detection=use_region_detection,
            format_block_content=format_block_content,
        )

    def to_dict(self) -> dict:
        return dict(
            use_doc_preprocessor=self.use_doc_preprocessor,
            use_seal_recognition=self.use_seal_recognition,
            use_table_recognition=self.use_table_recognition,
            use_formula_recognition=self.use_formula_recognition,
            use_chart_recognition=self.use_chart_recognition,
            use_region_detection=self.use_region_detection,
            format_block_content=self.format_block_content,
        )

    def validate(self) -> bool:
        """
        Check if the model settings are valid.

        Returns:
            bool: True if all required models are initialized according to settings, False otherwise.
        """
        return True


def validate_model_settings(input_params: dict, initialized_settings: dict) -> bool:
    """
    Check if the input parameters are valid based on the initialized models.

    Args:
        input_params (Dict): A dictionary containing input parameters.
        initialized_settings (Dict): A dictionary containing what models are initialized.

    Returns:
        bool: True if all required models are initialized according to input parameters, False otherwise.
    """

    if (
        input_params["use_doc_preprocessor"]
        and not initialized_settings["use_doc_preprocessor"]
    ):
        logging.error(
            "Set use_doc_preprocessor, but the models for doc preprocessor are not initialized.",
        )
        return False

    if (
        input_params["use_seal_recognition"]
        and not initialized_settings["use_seal_recognition"]
    ):
        logging.error(
            "Set use_seal_recognition, but the models for seal recognition are not initialized.",
        )
        return False

    if (
        input_params["use_table_recognition"]
        and not initialized_settings["use_table_recognition"]
    ):
        logging.error(
            "Set use_table_recognition, but the models for table recognition are not initialized.",
        )
        return False

    return True

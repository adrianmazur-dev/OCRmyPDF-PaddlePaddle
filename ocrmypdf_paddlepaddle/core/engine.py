from __future__ import annotations
import os

import cv2 as cv
import contextlib
import logging
import sys
import threading
import traceback
from typing import Optional, Tuple

from paddleocr import PPStructureV3, __version__ as paddleocr_version
import numpy.typing as npt
from ocrmypdf import OcrEngine

from ocrmypdf_paddlepaddle.config import ISO_639_3_2
from ocrmypdf_paddlepaddle.core.models import PaddleResult

try:
    import billiard as multiprocessing
except ImportError:
    import multiprocessing

logger = logging.getLogger(__name__)


Task = Tuple[npt.NDArray, multiprocessing.Value, threading.Event] | None


def ocr_process(q: multiprocessing.Queue[Task], options):
    reader: Optional[PPStructureV3] = None

    while True:
        message = q.get()
        if message is None:
            return  # exit process
        img, output_dict, event = message

        try:
            os.environ.pop("KMP_DEVICE_THREAD_LIMIT", None)
            os.environ.pop("KMP_TEAMS_THREAD_LIMIT", None)
            os.environ.pop("KMP_TEAMS_THREAD_LIMIT", None)
            os.environ.pop("OMP_THREAD_LIMIT", None)
            os.environ["OMP_NUM_THREADS"] = "1"

            if reader is None:
                languages = [ISO_639_3_2[lang] for lang in options.languages]

                # Build base kwargs for PPStructureV3
                paddle_kwargs = {
                    "lang": languages[0],
                    "use_table_recognition": True,
                    "use_chart_recognition": False,
                    "use_formula_recognition": False,
                    "use_seal_recognition": False,
                }

                # Merge with custom model config
                paddle_config = getattr(options, "_paddle_config", None)
                if paddle_config:
                    paddle_kwargs.update(paddle_config.to_ppstructure_kwargs())

                with contextlib.redirect_stdout(sys.stderr):
                    reader = PPStructureV3(**paddle_kwargs)

            results = reader.predict(
                input=img,
                use_table_recognition=True,
                use_seal_recognition=False,
                use_formula_recognition=False,
                use_chart_recognition=False,
            )
            output_dict["output"] = [
                PaddleResult.from_layout_result(result) for result in results
            ]
        except Exception as e:
            traceback.print_exception(e)
            output_dict["output"] = ""
        finally:
            event.set()


class PaddlePaddleEngine(OcrEngine):
    @staticmethod
    def version():
        return paddleocr_version

    @staticmethod
    def creator_tag(options):
        tag = "-PDF" if options.pdf_renderer == "sandwich" else ""
        return f"PaddlePaddle{tag} {PaddlePaddleEngine.version()}"

    def __str__(self):
        return f"PaddlePaddle {PaddlePaddleEngine.version()}"

    @staticmethod
    def languages(options):
        return ISO_639_3_2.keys()

    @staticmethod
    def get_orientation(input_file, options):
        raise NotImplementedError(
            "PaddlePaddle does not support orientation detection."
        )

    @staticmethod
    def get_deskew(input_file, options) -> float:
        raise NotImplementedError("PaddlePaddle does not support deskewing.")

    @staticmethod
    def generate_hocr(input_file, output_hocr, output_text, options):
        raise NotImplementedError("PaddlePaddle does not support hOCR output.")

    @staticmethod
    def generate_pdf(input_file, output_pdf, output_text, options):
        from ocrmypdf_paddlepaddle.generators.pdf import paddleocr_to_pdf

        img = cv.imread(os.fspath(input_file))
        if img is None:
            raise RuntimeError(f"Failed to load image: {input_file}")

        sync_data = options._engine_struct
        manager: multiprocessing.managers.SyncManager = sync_data["manager"]
        queue: multiprocessing.Queue[Task] = sync_data["queue"]
        output_dict = manager.dict()
        event = manager.Event()
        queue.put((img, output_dict, event))
        event.wait()

        result = output_dict["output"][0]

        paddleocr_to_pdf(
            image_filename=input_file,
            image_scale=1.0,
            ocr_result=result,
            output_pdf=output_pdf,
            boxes=True,
        )

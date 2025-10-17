from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pluggy
from ocrmypdf import Executor, PdfContext, hookimpl
from ocrmypdf.builtin_plugins.optimize import optimize_pdf as default_optimize_pdf

from ocrmypdf_paddlepaddle.logging import setup_logging
from ocrmypdf_paddlepaddle.core.engine import ocr_process, PaddlePaddleEngine
from ocrmypdf_paddlepaddle.config import PaddleConfig

setup_logging("DEBUG", "text")

try:
    import billiard as multiprocessing
except ImportError:
    import multiprocessing


class ProcessList:
    def __init__(self, plist):
        self.process_list = plist

    def __getstate__(self):
        return []


@hookimpl
def initialize(plugin_manager: pluggy.PluginManager):
    pass


@hookimpl
def add_options(parser):
    options = parser.add_argument_group("PaddlePaddle", "PaddlePaddle options")
    options.add_argument("--engine-workers", type=int, default=1)

    options.add_argument(
        "--paddle-config",
        type=str,
        default=None,
    )

    options.add_argument(
        "--text-detection-model-name",
        type=str,
        default=None,
    )
    options.add_argument(
        "--text-detection-model-dir",
        type=str,
        default=None,
    )

    options.add_argument(
        "--text-recognition-model-name",
        type=str,
        default=None,
    )
    options.add_argument(
        "--text-recognition-model-dir",
        type=str,
        default=None,
    )


@hookimpl
def check_options(options):
    options._paddle_config = PaddleConfig.from_dict(options.__dict__)

    manager = multiprocessing.Manager()
    queue = multiprocessing.Queue(-1)

    ocr_process_list = []
    for _ in range(options.engine_workers):
        task = multiprocessing.Process(
            target=ocr_process, args=(queue, options), daemon=True
        )
        task.start()
        ocr_process_list.append(task)

    options._engine_struct = {"manager": manager, "queue": queue}
    options._engine_processlist = ProcessList(ocr_process_list)


@hookimpl
def optimize_pdf(
    input_pdf: Path,
    output_pdf: Path,
    context: PdfContext,
    executor: Executor,
    linearize: bool,
) -> tuple[Path, Sequence[str]]:
    options = context.options

    for _ in range(options.engine_workers):
        q = options._engine_struct["queue"]
        q.put(None)  # send stop message
    for p in options._engine_processlist.process_list:
        p.join(3.0)  # clean up child processes but don't wait forever

    return default_optimize_pdf(
        input_pdf=input_pdf,
        output_pdf=output_pdf,
        context=context,
        executor=executor,
        linearize=linearize,
    )


@hookimpl
def get_ocr_engine():
    return PaddlePaddleEngine()

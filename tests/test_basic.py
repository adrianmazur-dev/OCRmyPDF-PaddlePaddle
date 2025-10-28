import ocrmypdf
import pikepdf
import pytest
from paddlex import create_pipeline

import ocrmypdf_paddlepaddle

FILE_NAME = "jbig2.pdf"


def test_ocr(pipeline_config_path, resources, output_resources):
    pipeline = create_pipeline(str(pipeline_config_path))
    output = pipeline.predict(
        input=str(resources / f"./{FILE_NAME}"),
    )
    for res in output:
        res.save_to_img(output_resources)
        res.save_to_json(output_resources)


# def test_plugin(resources, outpdf):
#     ocrmypdf.ocr(
#         resources / f"./{FILE_NAME}",
#         outpdf,
#         image_dpi=300,
#         pdf_renderer="sandwich",
#         force_ocr=True,
#     )
#     assert outpdf.exists()

#     with pikepdf.open(outpdf) as pdf:
#         assert "PaddlePaddle" in str(pdf.docinfo["/Creator"])

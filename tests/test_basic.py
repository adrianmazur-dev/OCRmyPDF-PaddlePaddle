import ocrmypdf
import pikepdf
import pytest

import ocrmypdf_paddlepaddle


def test_easyocr(resources, outpdf):
    ocrmypdf.ocr(resources / "jbig2.pdf", outpdf, pdf_renderer="sandwich")
    assert outpdf.exists()

    with pikepdf.open(outpdf) as pdf:
        assert "PaddlePaddle" in str(pdf.docinfo["/Creator"])

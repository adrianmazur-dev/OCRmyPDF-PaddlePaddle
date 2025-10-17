from __future__ import annotations

import importlib.resources
import logging
from math import atan2, cos, hypot, sin
from pathlib import Path

from pikepdf import (
    ContentStreamInstruction,
    Dictionary,
    Name,
    Operator,
    Pdf,
    unparse_content_stream,
)
from PIL import Image

from ocrmypdf_paddlepaddle.core.models import PaddleResult

log = logging.getLogger(__name__)
GLYPHLESS_FONT = importlib.resources.read_binary(
    "ocrmypdf_paddlepaddle.resources.fonts", "pdf.ttf"
)

# PDF generation constants
CHAR_ASPECT = 2  # Character aspect ratio for glyphless font (width:height = 1:2)
ANGLE_THRESHOLD_RAD = (
    0.01  # Angle threshold in radians (~0.57°) for considering text horizontal
)
TEXT_RENDER_MODE_INVISIBLE = 3  # PDF text rendering mode for invisible text
PDF_POINTS_PER_INCH = 72.0  # Standard PDF conversion: 72 points = 1 inch
HORIZONTAL_SCALE_FACTOR = 100.0  # Base percentage for PDF horizontal text scaling


def pt_from_pixel(bbox, scale, height):
    """Convert pixel coordinates to PDF points."""
    point_pairs = [
        (x * scale[0], (height - y) * scale[1]) for x, y in zip(bbox[0::2], bbox[1::2])
    ]
    return [elm for pt in point_pairs for elm in pt]


def bbox_to_poly(bbox):
    """Convert bbox [x1, y1, x2, y2] to polygon format [[x0,y0], [x1,y1], [x2,y2], [x3,y3]]."""
    x1, y1, x2, y2 = bbox
    # Return corners in order: top-left, top-right, bottom-right, bottom-left
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def poly_to_quad(poly):
    """Convert PaddlePaddle polygon to quad format (x0,y0,x1,y1,x2,y2,x3,y3)."""
    # PaddlePaddle format: [[x0,y0], [x1,y1], [x2,y2], [x3,y3]]
    # Quad format: [x0,y0, x1,y1, x2,y2, x3,y3]
    return [coord for point in poly for coord in point]


def register_glyphlessfont(pdf: Pdf):
    PLACEHOLDER = Name.Placeholder

    basefont = pdf.make_indirect(
        Dictionary(
            BaseFont=Name.GlyphLessFont,
            DescendantFonts=[PLACEHOLDER],
            Encoding=Name("/Identity-H"),
            Subtype=Name.Type0,
            ToUnicode=PLACEHOLDER,
            Type=Name.Font,
        )
    )
    cid_font_type2 = pdf.make_indirect(
        Dictionary(
            BaseFont=Name.GlyphLessFont,
            CIDToGIDMap=PLACEHOLDER,
            CIDSystemInfo=Dictionary(
                Ordering="Identity",
                Registry="Adobe",
                Supplement=0,
            ),
            FontDescriptor=PLACEHOLDER,
            Subtype=Name.CIDFontType2,
            Type=Name.Font,
            DW=1000 // CHAR_ASPECT,
        )
    )
    basefont.DescendantFonts = [cid_font_type2]
    cid_font_type2.CIDToGIDMap = pdf.make_stream(b"\x00\x01" * 65536)
    basefont.ToUnicode = pdf.make_stream(
        b"/CIDInit /ProcSet findresource begin\n"
        b"12 dict begin\n"
        b"begincmap\n"
        b"/CIDSystemInfo\n"
        b"<<\n"
        b"  /Registry (Adobe)\n"
        b"  /Ordering (UCS)\n"
        b"  /Supplement 0\n"
        b">> def\n"
        b"/CMapName /Adobe-Identify-UCS def\n"
        b"/CMapType 2 def\n"
        b"1 begincodespacerange\n"
        b"<0000> <FFFF>\n"
        b"endcodespacerange\n"
        b"1 beginbfrange\n"
        b"<0000> <FFFF> <0000>\n"
        b"endbfrange\n"
        b"endcmap\n"
        b"CMapName currentdict /CMap defineresource pop\n"
        b"end\n"
        b"end\n"
    )
    font_descriptor = pdf.make_indirect(
        Dictionary(
            Ascent=1000,
            CapHeight=1000,
            Descent=-1,
            Flags=5,  # Fixed pitch and symbolic
            FontBBox=[0, 0, 1000 // CHAR_ASPECT, 1000],
            FontFile2=PLACEHOLDER,
            FontName=Name.GlyphLessFont,
            ItalicAngle=0,
            StemV=80,
            Type=Name.FontDescriptor,
        )
    )
    font_descriptor.FontFile2 = pdf.make_stream(GLYPHLESS_FONT)
    cid_font_type2.FontDescriptor = font_descriptor
    return basefont


class ContentStreamBuilder:
    """Builder for PDF content stream instructions.

    This is a mutable builder that efficiently constructs PDF content streams
    using method chaining. Each operation modifies the builder in-place and
    returns self for chaining. Call build() to get the final instruction list.

    Example:
        cs = ContentStreamBuilder()
        cs.q().BT().Tf(Name.F1, 12).TJ("Hello").ET().Q()
        instructions = cs.build()
    """

    def __init__(self):
        self._instructions = []

    def q(self):
        """Save the graphics state."""
        self._instructions.append(ContentStreamInstruction([], Operator("q")))
        return self

    def Q(self):
        """Restore the graphics state."""
        self._instructions.append(ContentStreamInstruction([], Operator("Q")))
        return self

    def cm(self, a: float, b: float, c: float, d: float, e: float, f: float):
        """Concatenate matrix."""
        self._instructions.append(
            ContentStreamInstruction([a, b, c, d, e, f], Operator("cm"))
        )
        return self

    def BT(self):
        """Begin text object."""
        self._instructions.append(ContentStreamInstruction([], Operator("BT")))
        return self

    def ET(self):
        """End text object."""
        self._instructions.append(ContentStreamInstruction([], Operator("ET")))
        return self

    def BDC(self, mctype: Name, mcid: int):
        """Begin marked content sequence."""
        self._instructions.append(
            ContentStreamInstruction([mctype, Dictionary(MCID=mcid)], Operator("BDC"))
        )
        return self

    def EMC(self):
        """End marked content sequence."""
        self._instructions.append(ContentStreamInstruction([], Operator("EMC")))
        return self

    def Tf(self, font: Name, size: int):
        """Set text font and size."""
        self._instructions.append(
            ContentStreamInstruction([font, size], Operator("Tf"))
        )
        return self

    def Tm(self, a: float, b: float, c: float, d: float, e: float, f: float):
        """Set text matrix."""
        self._instructions.append(
            ContentStreamInstruction([a, b, c, d, e, f], Operator("Tm"))
        )
        return self

    def Tr(self, mode: int):
        """Set text rendering mode."""
        self._instructions.append(ContentStreamInstruction([mode], Operator("Tr")))
        return self

    def Tz(self, scale: float):
        """Set text horizontal scaling."""
        self._instructions.append(ContentStreamInstruction([scale], Operator("Tz")))
        return self

    def TJ(self, text):
        """Show text."""
        self._instructions.append(
            ContentStreamInstruction([[text.encode("utf-16be")]], Operator("TJ"))
        )
        return self

    def s(self):
        """Stroke and close path."""
        self._instructions.append(ContentStreamInstruction([], Operator("s")))
        return self

    def re(self, x: float, y: float, w: float, h: float):
        """Append rectangle to path."""
        self._instructions.append(
            ContentStreamInstruction([x, y, w, h], Operator("re"))
        )
        return self

    def RG(self, r: float, g: float, b: float):
        """Set RGB stroke color."""
        self._instructions.append(ContentStreamInstruction([r, g, b], Operator("RG")))
        return self

    def build(self):
        """Build and return instructions list."""
        return self._instructions


def generate_text_content_stream(
    ocr_result: PaddleResult,
    scale: tuple[float, float],
    height: int,
    boxes=False,
):
    cs = ContentStreamBuilder()
    cs.q()

    # Handle empty or None OCR results
    if not ocr_result or not ocr_result.blocks:
        cs.Q()
        return cs.build()

    # Use blocks to maintain proper reading order
    # For each block, use ocr_words to add text in order
    # MCID (Marked Content ID) counter for PDF accessibility - only incremented for
    mcid_counter = 0

    for block in ocr_result.blocks:
        # Get OCR words directly from block
        ocr_words = block.ocr_words

        if not ocr_words:
            log.debug(f"Block {block.label} has no ocr_words, skipping")
            continue

        # Add each word to the content stream
        for word_data in ocr_words:
            try:
                text = word_data.get("text", "")
                word_bbox = word_data.get("bbox", [])
                score = word_data.get("score", 0.0)

                # Skip empty text or invalid bbox
                if not text or len(word_bbox) != 4:
                    continue

                # Convert bbox to polygon format
                poly = bbox_to_poly(word_bbox)
                quad = poly_to_quad(poly)

                if len(quad) != 8:
                    log.warning(f"Invalid quad format, expected 8 coords: {quad}")
                    continue

                bbox = pt_from_pixel(quad, scale, height)
                angle = -atan2(bbox[5] - bbox[7], bbox[4] - bbox[6])
                if abs(angle) < ANGLE_THRESHOLD_RAD:
                    angle = 0.0
                cos_a, sin_a = cos(angle), sin(angle)

                font_size = hypot(bbox[0] - bbox[6], bbox[1] - bbox[7])

                # TODO: Implement proper space width calculation
                space_width = 0
                box_width = hypot(bbox[4] - bbox[6], bbox[5] - bbox[7]) + space_width

                # Validate computed values
                if len(text) == 0 or box_width <= 0 or font_size <= 0:
                    continue

                h_stretch = (
                    HORIZONTAL_SCALE_FACTOR
                    * box_width
                    / len(text)
                    / font_size
                    * CHAR_ASPECT
                )

                # Add invisible text layer
                cs.BT()
                cs.BDC(Name.Span, mcid_counter)
                cs.Tr(TEXT_RENDER_MODE_INVISIBLE)
                cs.Tm(cos_a, -sin_a, sin_a, cos_a, bbox[6], bbox[7])
                cs.Tf(Name("/f-0-0"), font_size)
                cs.Tz(h_stretch)
                cs.TJ(text)
                cs.EMC()
                cs.ET()

                # Add debug boxes if requested
                if boxes:
                    cs.q()
                    cs.cm(cos_a, -sin_a, sin_a, cos_a, bbox[6], bbox[7])
                    cs.re(0, 0, box_width, font_size)
                    cs.RG(100, 100, 100)
                    cs.s()
                    cs.Q()

                mcid_counter += 1
            except Exception as e:
                log.warning(f"Error processing word '{text[:50]}...': {e}")
                continue

    cs.Q()
    return cs.build()


def paddleocr_to_pdf(
    image_filename: Path,
    image_scale: float,
    ocr_result: PaddleResult,
    output_pdf: Path,
    boxes: bool = False,
):
    # Validate inputs
    if not image_filename.exists():
        raise ValueError(f"Image file not found: {image_filename}")

    if image_scale <= 0:
        raise ValueError(f"Invalid image_scale: {image_scale}. Must be positive.")

    # Open and validate image
    try:
        with Image.open(image_filename) as im:
            dpi = im.info.get("dpi", (PDF_POINTS_PER_INCH, PDF_POINTS_PER_INCH))
            width = im.width
            height = im.height

            if width <= 0 or height <= 0:
                raise ValueError(f"Invalid image dimensions: {width}x{height}")

            scale = (
                PDF_POINTS_PER_INCH / dpi[0] / image_scale,
                PDF_POINTS_PER_INCH / dpi[1] / image_scale,
            )
    except Exception as e:
        raise ValueError(f"Failed to open image {image_filename}: {e}") from e

    # Validate OCR results
    if not ocr_result or not ocr_result.blocks:
        log.warning(f"No OCR results found for {image_filename}, creating blank PDF")

    # Generate PDF
    try:
        with Pdf.new() as pdf:
            pdf.add_blank_page(page_size=(width * scale[0], height * scale[1]))
            pdf.pages[0].Resources = Dictionary(
                Font=Dictionary({"/f-0-0": register_glyphlessfont(pdf)})
            )

            cs = generate_text_content_stream(ocr_result, scale, height, boxes=boxes)
            pdf.pages[0].Contents = pdf.make_stream(unparse_content_stream(cs))

            pdf.save(output_pdf)
    except Exception as e:
        raise RuntimeError(f"Failed to generate PDF: {e}") from e

    return output_pdf

from paddlex.inference.pipelines.layout_parsing.layout_objects import LayoutBlock


class EnhancedLayoutBlock(LayoutBlock):
    def __init__(self, label: str, bbox: list, content: str = "", ocr_words: list = []):
        super().__init__(label, bbox, content)
        self.ocr_words = ocr_words

    def generate_ocr_blocks(self, rec_texts: list, rec_boxes: list, rec_scores: list):
        ocr_words = []
        for box_no in range(len(rec_texts)):
            ocr_words.append(
                {
                    "bbox": rec_boxes[box_no],
                    "text": rec_texts[box_no],
                    "score": rec_scores[box_no],
                }
            )
        self.ocr_words = ocr_words

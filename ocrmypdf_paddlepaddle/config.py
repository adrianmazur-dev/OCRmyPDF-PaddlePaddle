from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

ISO_639_3_2: dict[str, str] = {
    "afr": "af",
    "alb": "sq",
    "bel": "be",
    "bos": "bs",
    "ces": "cs",
    "chi_sim": "ch",
    "chi_tra": "chinese_cht",
    "cym": "cy",
    "cze": "cs",
    "dan": "da",
    "deu": "de",
    "dut": "nl",
    "eng": "en",
    "est": "et",
    "esp": "es",
    "fra": "fr",
    "gle": "ga",
    "hrv": "hr",
    "hun": "hu",
    "ice": "is",
    "ind": "id",
    "isl": "is",
    "ita": "it",
    "jpn": "japan",
    "kor": "korean",
    "lat": "la",
    "lit": "lt",
    "may": "ms",
    "msa": "ms",
    "nld": "nl",
    "nor": "no",
    "oci": "oc",
    "pol": "pl",
    "por": "pt",
    "rus": "ru",
    "slk": "sk",
    "slo": "sk",
    "slv": "sl",
    "spa": "es",
    "swa": "sw",
    "swe": "sv",
    "tha": "th",
    "tgl": "tl",
    "tur": "tr",
    "ukr": "uk",
}


@dataclass
class PaddleConfig:
    # Text detection model
    text_detection_model_name: Optional[str] = None
    text_detection_model_dir: Optional[str] = None

    # Text recognition model
    text_recognition_model_name: Optional[str] = None
    text_recognition_model_dir: Optional[str] = None

    def to_ppstructure_kwargs(self) -> dict:
        kwargs = {}
        for key, value in self.__dict__.items():
            if value is not None:
                kwargs[key] = value
        return kwargs

    @classmethod
    def from_dict(cls, data: dict) -> PaddleConfig:
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered_data)

    @classmethod
    def from_json_file(cls, path: Path | str) -> PaddleConfig:
        path = Path(path)
        with open(path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)

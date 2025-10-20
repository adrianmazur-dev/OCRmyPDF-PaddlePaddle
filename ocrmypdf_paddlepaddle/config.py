from __future__ import annotations
from pathlib import Path
from typing import Optional

PIPELINE_CONFIG_PATH = (
    Path(__file__).parent / "resources" / "configuration" / "pipeline-structure.yaml"
)


def load_pipeline_structure(custom_config_path: Optional[str] = None) -> str:
    if custom_config_path:
        config_path = Path(custom_config_path)
        if not config_path.exists():
            raise FileNotFoundError(
                f"Custom pipeline configuration file not found: {custom_config_path}"
            )
        return str(config_path.resolve())

    return str(PIPELINE_CONFIG_PATH)


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

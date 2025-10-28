from pathlib import Path

import pytest

TESTS_ROOT = Path(__file__).parent.resolve()
PROJECT_ROOT = TESTS_ROOT

PACKAGE_ROOT = PROJECT_ROOT.parent.resolve()


@pytest.fixture(scope="session")
def pipeline_config_path() -> Path:
    return (
        PACKAGE_ROOT
        / "ocrmypdf_paddlepaddle"
        / "resources"
        / "configuration"
        / "pipeline-structure.yaml"
    )


@pytest.fixture(scope="session")
def resources() -> Path:
    return Path(TESTS_ROOT) / "resources"


@pytest.fixture(scope="session")
def output_resources() -> Path:
    return Path(TESTS_ROOT) / "output_resources"


@pytest.fixture(scope="function")
def outdir(tmp_path) -> Path:
    return tmp_path


@pytest.fixture(scope="function")
def outpdf(tmp_path) -> Path:
    return tmp_path / "out.pdf"

from pathlib import Path

import pytest

SKFIN_ROOT = Path(__file__).resolve().parent.parent
SKFIN_NBS_DIR = SKFIN_ROOT / "nbs"


def pytest_addoption(parser):
    parser.addoption(
        "--nb-timeout",
        action="store",
        default=600,
        type=int,
        help="Timeout in seconds for notebook execution",
    )
    parser.addoption(
        "--run-network",
        action="store_true",
        default=False,
        help="Run tests that require internet access",
    )


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-network"):
        skip_network = pytest.mark.skip(reason="needs --run-network to run")
        for item in items:
            if "network" in item.keywords:
                item.add_marker(skip_network)


@pytest.fixture
def nb_timeout(request):
    return request.config.getoption("--nb-timeout")

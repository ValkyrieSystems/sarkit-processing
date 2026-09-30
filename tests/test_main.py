import subprocess
import sys

import pytest

import sarkit_processing


def ep_callable(epname):
    def func(args):
        return subprocess.check_output([epname] + args, text=True)

    return func


def run_module(args):
    return subprocess.check_output(
        [sys.executable, "-m", "sarkit_processing"] + args, text=True
    )


@pytest.mark.parametrize(
    "caller", [ep_callable("sarkit-processing"), ep_callable("skp"), run_module]
)
def test_main(caller):
    out = caller(["--version"])
    assert sarkit_processing.__version__ in out
    assert "sarkit-processing" in out

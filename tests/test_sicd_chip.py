import argparse
import math
import subprocess
import sys

import pytest
import sarkit.sicd as sksicd
import shapely.geometry as shg

import sarkit_processing.__main__
import sarkit_processing._sicd_chip as spsc
import tests.utils


def _check_file(filename, bounds):
    expected_nrows = math.ceil(bounds[2]) - math.floor(bounds[0]) + 1
    expected_ncols = math.ceil(bounds[3]) - math.floor(bounds[1]) + 1
    with filename.open("rb") as file, sksicd.NitfReader(file) as reader:
        ew = sksicd.ElementWrapper(reader.metadata.xmltree.getroot())
        assert ew["ImageData"]["FirstRow"] == math.floor(bounds[0])
        assert ew["ImageData"]["FirstCol"] == math.floor(bounds[1])
        assert ew["ImageData"]["NumRows"] == expected_nrows
        assert ew["ImageData"]["NumCols"] == expected_ncols


def test_wkt(example_sicd, tmp_path):

    geometry = shg.box(10, 20, 60, 120).buffer(1.1)

    wkt_str = geometry.wkt
    wkt_file = tmp_path / "coords.wkt"
    wkt_file.write_text(wkt_str)

    chip_file = tmp_path / "chip_wkt_cli.sicd"
    sarkit_processing.__main__.main(
        ["sicd_chip", str(example_sicd), str(chip_file), wkt_str]
    )
    _check_file(chip_file, geometry.bounds)

    chip_file = tmp_path / "file_wkt_file.sicd"
    sarkit_processing.__main__.main(
        ["sicd_chip", str(example_sicd), str(chip_file), str(wkt_file)]
    )
    _check_file(chip_file, geometry.bounds)

    chip_file = tmp_path / "stdin_wkt_cli.sicd"
    subprocess.run(
        [
            "sarkit-processing",
            "sicd_chip",
            example_sicd,
            chip_file,
            "-",
        ],
        input=wkt_str.encode(),
        check=True,
    )
    _check_file(chip_file, geometry.bounds)


_BOUNDS = [10.1, 20.1, 59.9, 119.9]


@pytest.mark.parametrize(
    "bounds_str",
    [
        " ".join([str(val) for val in _BOUNDS]),
        ", ".join([str(val) for val in _BOUNDS]),
        ",".join([str(val) for val in _BOUNDS]),
    ],
)
def test_bounds(example_sicd, tmp_path, bounds_str):

    bounds_file = tmp_path / "bounds.txt"
    bounds_file.write_text(bounds_str)

    chip_file = tmp_path / "chip_cli.sicd"
    sarkit_processing.__main__.main(
        ["sicd_chip", str(example_sicd), str(chip_file), bounds_str]
    )
    _check_file(chip_file, _BOUNDS)

    chip_file = tmp_path / "file_wkt_cli.sicd"
    sarkit_processing.__main__.main(
        ["sicd_chip", str(example_sicd), str(chip_file), str(bounds_file)]
    )
    _check_file(chip_file, _BOUNDS)

    chip_file = tmp_path / "stdin_wkt_cli.sicd"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "sarkit_processing",
            "sicd_chip",
            example_sicd,
            chip_file,
            "-",
        ],
        input=bounds_str.encode(),
        check=True,
    )
    _check_file(chip_file, _BOUNDS)


def test_edges(example_sicd, tmp_path):
    with example_sicd.open("rb") as file, sksicd.NitfReader(file) as reader:
        ew = sksicd.ElementWrapper(reader.metadata.xmltree.getroot())
        nrows = ew["ImageData"]["NumRows"]
        ncols = ew["ImageData"]["NumCols"]

    chip_file = tmp_path / "chip.sicd"
    sarkit_processing.__main__.main(
        ["sicd_chip", str(example_sicd), str(chip_file), "-10 -10 10 10"]
    )
    _check_file(chip_file, [0, 0, 10, 10])

    sarkit_processing.__main__.main(
        [
            "sicd_chip",
            str(example_sicd),
            str(chip_file),
            f"{nrows - 10} {ncols - 10} {nrows + 10} {ncols + 10}",
        ]
    )
    _check_file(chip_file, [nrows - 10, ncols - 10, nrows - 1, ncols - 1])


@pytest.mark.parametrize(
    "bbox",
    ["-10 -10 -1 -1", "-10 -10 -2 -2", "10 10 -2 -2", "100000 100000 200000 200000"],
)
def test_invalid(example_sicd, tmp_path, bbox):
    chip_file = tmp_path / "chip.sicd"
    with pytest.raises(ValueError, match="must be before"):
        sarkit_processing.__main__.main(
            ["sicd_chip", str(example_sicd), str(chip_file), bbox]
        )


def test_empty(example_sicd, tmp_path):
    chip_file = tmp_path / "chip.sicd"
    with pytest.raises(RuntimeError, match="No coordinates provided"):
        sarkit_processing.__main__.main(
            ["sicd_chip", str(example_sicd), str(chip_file), "POINT EMPTY"]
        )


def test_bounds_multi_arg_cli(example_sicd, tmp_path):
    chip_file = tmp_path / "chip_cli.sicd"
    sarkit_processing.__main__.main(
        ["sicd_chip", str(example_sicd), str(chip_file)] + [str(val) for val in _BOUNDS]
    )
    _check_file(chip_file, _BOUNDS)


def test_smart_open(example_sicd, tmp_path):
    chip_file = tmp_path / "chip_cli.sicd"
    with tests.utils.static_http_server(example_sicd.parent) as server_url:
        sarkit_processing.__main__.main(
            ["sicd_chip", f"{server_url}/{example_sicd.name}", str(chip_file)]
            + [str(val) for val in _BOUNDS]
        )
    _check_file(chip_file, _BOUNDS)


def test_subcommand(example_sicd, tmp_path):
    sc = spsc.SicdChipSubcommand()
    parser = argparse.ArgumentParser(**sc.get_argument_parser_kwargs())
    sc.add_arguments(parser)

    chip_file = tmp_path / "chip.sicd"
    config = parser.parse_args(
        [str(example_sicd), str(chip_file)] + [str(val) for val in _BOUNDS]
    )
    assert sc.run_command(config) == 0
    assert chip_file.stat().st_size < example_sicd.stat().st_size

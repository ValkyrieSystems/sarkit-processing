import copy
import math
import sys

import numpy as np
import sarkit.sicd as sksicd
import shapely

from sarkit_processing import _cli

try:
    from smart_open import open
except ImportError:
    pass


class SicdChipSubcommand(_cli.Subcommand):
    def get_argument_parser_kwargs(self):
        return dict(
            description="Create SICD sub-image",
        )

    def add_arguments(self, parser):
        parser.add_argument("input_sicd_file", help="Path to input SICD file")
        parser.add_argument("output_sicd_file", help="Path of SICD file to write")
        parser.add_argument(
            "coordinates",
            nargs="+",
            help="WKT containing (row, col) or 4 value bounding box [first_row, first_col, last_row, last_col]. May be filename or string. '-' reads from stdin",
        )

    def run_command(self, config):
        if config.coordinates[0] == "-":
            coordinates = sys.stdin.read()
        else:
            try:
                with open(config.coordinates[0], "r") as file:
                    coordinates = file.read()
            except (OSError, FileNotFoundError):
                coordinates = " ".join(config.coordinates)

        try:
            geometry = shapely.from_wkt(coordinates)
            if shapely.get_num_coordinates(geometry) == 0:
                # more helpful error message than NaNs in the bounds
                raise RuntimeError("No coordinates provided")
            bounds = geometry.bounds
        except shapely.errors.GEOSException:
            sep = "," if "," in coordinates else " "
            bounds = np.fromstring(coordinates, np.float64, count=4, sep=sep)

        with (
            open(config.input_sicd_file, "rb") as file,
            sksicd.NitfReader(file) as reader,
        ):
            xmltree = reader.metadata.xmltree
            ew = sksicd.ElementWrapper(xmltree.getroot())

            start_row = max(0, int(math.floor(bounds[0])))
            start_col = max(0, int(math.floor(bounds[1])))
            stop_row = min(ew["ImageData"]["NumRows"], int(math.ceil(bounds[2])) + 1)
            stop_col = min(ew["ImageData"]["NumCols"], int(math.ceil(bounds[3])) + 1)

            if stop_row <= start_row or stop_col <= start_col:
                raise ValueError(
                    f"start index {start_row, start_col} must be before stop index {stop_row, stop_col}"
                )

            subimage_data, subimage_xmltree = reader.read_sub_image(
                start_row, start_col, stop_row, stop_col
            )
            metadata = copy.deepcopy(reader.metadata)
            metadata.xmltree = subimage_xmltree
            with (
                open(config.output_sicd_file, "wb") as file,
                sksicd.NitfWriter(file, metadata) as writer,
            ):
                writer.write_image(subimage_data)

        return 0

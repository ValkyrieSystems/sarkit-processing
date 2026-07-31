import argparse
import re
import typing

NEGATIVE_NUMBER_EXPONENTAL_MATCHER = re.compile(r"^-(\d+\.?\d*|\.\d+)([eE][+\-]?\d+)?$")


def allow_floating_point_arguments(parser):
    parser._negative_number_matcher = NEGATIVE_NUMBER_EXPONENTAL_MATCHER


class Subcommand:
    """Class describing a CLI subcommand"""

    def get_argument_parser_kwargs(self) -> dict[str, typing.Any]:
        """ArgumentParser constructor arguments

        Returns
        -------
        dict
            dictionary of ArgumentParser arguments
        """
        raise NotImplementedError()

    def add_arguments(self, parser: argparse.ArgumentParser):
        """Add arguments to a parser

        Parameters
        ----------
        parser : argparse.ArgumentParser

        Returns
        -------
        None
        """
        raise NotImplementedError()

    def run_command(self, config: argparse.Namespace) -> int:
        """Run the subcommand

        Parameters
        ----------
        config : argparse.Namespace

        Returns
        -------
        int
            return code
        """
        raise NotImplementedError()

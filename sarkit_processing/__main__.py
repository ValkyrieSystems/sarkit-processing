import argparse
import sys

import sarkit_processing._coords
import sarkit_processing._sicd_chip
from sarkit_processing import _cli


def main(args=None):
    parser = argparse.ArgumentParser(description="sarkit-processing tools")
    subcommands = parser.add_subparsers(
        title="subcommands", required=True, dest="command"
    )

    def add_subcommand(name, sc: _cli.Subcommand) -> None:
        kwargs = sc.get_argument_parser_kwargs()
        if "help" not in kwargs:
            # "help" on subparsers displays on their parent's help
            kwargs["help"] = kwargs["description"]
        command = subcommands.add_parser(name, **kwargs)
        sc.add_arguments(command)
        command.set_defaults(command_handler=sc.run_command)

    add_subcommand("coords", sarkit_processing._coords.CoordsSubcommand())
    add_subcommand("sicd_chip", sarkit_processing._sicd_chip.SicdChipSubcommand())

    config = parser.parse_args(args)
    return config.command_handler(config)


if __name__ == "__main__":
    sys.exit(main())

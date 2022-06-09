import argparse
from sys import argv

parser = argparse.ArgumentParser("Land Cover Projection Shifter")

parser.add_argument(
    "--tile-size", "-ts",
    dest='ts',
    type=int,
    default=1024,
    help="Size of tiles to process."
)
parser.add_argument(
    "-dsm",
    type=str,
    help="Path to DSM.",
    required=True
)
parser.add_argument(
    "-lcm",
    type=str,
    help="Path to land cover map.",
    required=True
)

def pair(arg):
    return [int(x) for x in arg.split(',')];

parser.add_argument(
    "-a",
    help="Sequence of pairs of angles to produce.",
    dest='angles',
    type=pair,
    required=True,
    nargs="+"
)


args = parser.parse_args(argv[1:])
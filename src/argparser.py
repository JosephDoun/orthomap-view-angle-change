import argparse
from sys import argv

parser = argparse.ArgumentParser("Projector")

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
    help="Path to Digital Surface Model.",
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
    metavar="<azimuth,zenith>",
    type=pair,
    required=True,
    nargs="+"
)
parser.add_argument(
    "-c", dest='c',
    type=int,
    metavar="<num_cores>",
    default=4,
    # nargs=1
)
parser.add_argument(
    "-t", dest='threads',
    type=int,
    metavar='<num_threads>',
    default=3,
    nargs=1
)

args = parser.parse_args(argv[1:])

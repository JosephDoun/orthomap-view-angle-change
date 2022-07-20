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
    default=8,
    # nargs=1
)
parser.add_argument(
    "-t", dest='threads',
    type=int,
    metavar='<num_threads>',
    default=4,
    nargs=1
)
parser.add_argument(
    "--target-res",
    "-tr",
    dest='tr',
    help="""
    Target resolution for X and Y in meters.
    Defaults to 100 meters.
    """,
    type=float,
    default=100.0
)
parser.add_argument(
    "--nogo",
    dest="nogo",
    action="store_true",
    help="""
    Do not produce the final rescaled product.
    
    Use switch if there are memory issues. Constructing the final product
    requires the entire map to fit in memory.
    
    This is a switch to deactivate this process.
    """
)

args = parser.parse_args(argv[1:])

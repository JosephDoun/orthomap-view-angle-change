import argparse
from sys import argv

parser = argparse.ArgumentParser("Projector")

parser.add_argument(
    "--tile-size", "-ts",
    dest='ts',
    type=int,
    default=1000,
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

def spair(arg):
    return [float(x) for x in arg.split(',')];

def rpair(arg):
    return [str(x) for x in arg.split(',')]

parser.add_argument(
    "--scalar-angles", "-sa",
    help="Sequence of pairs of scalar angles for processing.",
    dest='angles',
    metavar="azimuth,zenith",
    type=spair,
    required=False,
    nargs="+",
    default=[]
)
parser.add_argument(
    "--raster-angles", "-ra",
    help="""
    Sequence of pairs of angle rasters for processing.
    
    Rasters have to have the same dimensions/resolution
    as the other inputs. (1m spatial resolution)
    
    NOT IMPLEMENTED/TESTED AT THE MOMENT. (Hi David)
    """,
    dest="rangles",
    metavar="azim_band_path,zen_band_path",
    type=rpair,
    nargs='+',
    default=[],
    required=False
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

if not args.angles and not args.rangles:
    raise AssertionError("""
                         
        Error: One of the following arguments is required:
        
        --scalar-angles/-sa, --raster-angles/-ra
        
        """
    )

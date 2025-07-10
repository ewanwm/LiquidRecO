"""LiquidReco main app.

Takes a root file that contains 2D fiber hits, creates an EventProcessor
based on the provided arguments, and processes the fiber hits using this.

:raises ValueError: if input args are not valid
"""

import argparse
import sys

from liquidreco.event_processor import EventProcessor

def main():

    ## Set up the arg parser instance
    arg_parser = argparse.ArgumentParser(description=__doc__)

    ## options related to the input
    base_args = arg_parser.add_argument_group("Base options")
    base_args.add_argument("--input-file", "-i", help = "The input file containing hits to be reconstructed")
    base_args.add_argument("--tree-name", "-t", type=str,
        help = "The name of the tree in the input file that contains the 2D fiber hits.")
    base_args.add_argument("--max-n-events", "-n", default=None, type=int, required=False,
        help = "If set, will only process the first <n> events")

    ## options related to the construction of the hits
    hit_args = arg_parser.add_argument_group("Hit related options")
    hit_args.add_argument(
        "--weight-scaling", type=float, default=None, 
        help="Apply a weight scaling to events. Might be useful if e.g. you want to compare truth hits to hits with DAQ simulation or fiber effects.", 
        required=False)
    hit_args.add_argument("--normalise", action='store_true', required=False)
    hit_args.add_argument("--norm-window-size", type=int, default=3, required=False)
    hit_args.add_argument("--find-2d-peaks", action='store_true', required=False,
        help = "Try to find peaks in the 2D fiber hit projections before building 3D hits")
    hit_args.add_argument("--allow-2-fiber-hits", action='store_true', required=False,
        help = "Allow 3D hits to be made from 2 (XZ and YZ) fibers instead of also requiring an XY fiber")
    hit_args.add_argument("--min-2d-hit-weight", default=0.0, type=float, required=False,
        help = "Minimum weight required for a 2D fiber hit to be used to build 3D hits")
    hit_args.add_argument("--n-required-peak-hits", default=None, type=float, required=False,
        help = "minimum number of 'peak' hits in each direction required for a valid 3D hit")
    hit_args.add_argument("--fit-algorithm", default=None, type=str, required=False,
        help = "The name of the algorithm to use to fit tracks")

    ## options related to plotting 
    plotting_args = arg_parser.add_argument_group("Plotting related options")
    plotting_args.add_argument("--make-gifs", action='store_true', required=False)
    plotting_args.add_argument("--make-fiber-hit-plots", action='store_true', required=False)
    plotting_args.add_argument("--make-3d-hit-plots", action='store_true', required=False)
    plotting_args.add_argument("--make-debug-plots", action='store_true', required=False)
    plotting_args.add_argument("--charge-cutoff", default=100.0, type=float, required=False)
    plotting_args.add_argument("--gif-timesteps", default=1.0, type=float, required=False)

    # args to define geometry
    geom_args = arg_parser.add_argument_group("Geometry related arguments")
    geom_args.add_argument("--x-fiber-x-pos", default=0.0, type=float, required=False, 
                            help="any fiber hit with this x value is considered to be an x fibr hit")
    geom_args.add_argument("--y-fiber-y-pos", default=30.0, type=float, required=False, 
                            help="any fiber hit with this y value is considered to be a y fibr hit")
    geom_args.add_argument("--z-fiber-z-pos", default=910.0, type=float, required=False, 
                            help="any fiber hit with this z value is considered to be a z fibr hit")

    geom_args.add_argument("--plot-centre-x", default=0.0, type=float, required=False)
    geom_args.add_argument("--plot-centre-y", default=30.0, type=float, required=False)
    geom_args.add_argument("--plot-centre-z", default=910.0, type=float, required=False)

    geom_args.add_argument("--plot-size-x", default=2400.0, type=float, required=False)
    geom_args.add_argument("--plot-size-y", default=2400.0, type=float, required=False)
    geom_args.add_argument("--plot-size-z", default=1300.0, type=float, required=False)

    args = arg_parser.parse_args(sys.argv[1:])

    # some checks on validity of arguments
    if args.norm_window_size %2 == 0:
        raise ValueError("norm-window-size must be odd")

    processor = EventProcessor(args, args.input_file, args.make_fiber_hit_plots, args.make_3d_hit_plots)

    processor.read_hit_info(args.tree_name)

    if args.weight_scaling is not None:
        processor.apply_weight_scaling(args.weight_scaling)

    processor.process()

    processor.finalise()

if __name__ == "__main__":
    main()

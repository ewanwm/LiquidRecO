
import numpy as np
import argparse
import sys

from liquidreco.event_processor import EventProcessor

arg_parser = argparse.ArgumentParser(
    "Simple 3D Homo FGD Reco"
)

## options related to the input
arg_parser.add_argument("--input-file", "-i", help = "The input file containing hits to be reconstructed")
arg_parser.add_argument("--use-true-hits", action='store_true', required=False, 
    help = "Use truth hits. These are just Geant4 level truth hits and include all light that falls on a fiber, with no attenuation, light trapping, sensor or electronics simulation.")
arg_parser.add_argument("--use-raw-hits", action='store_true', required=False,
    help = "Use the 'raw' hits: these are geant4 hits with fiber attenuation and light trapping simulated, but no sensors or electronincs.")
arg_parser.add_argument(
    "--no-truth-scaling", action='store_true', 
    help="*don't* apply downscaling to the true hits (which roughly accounts for fiber attenuation, photon trapping prob., sensor efficiency etc.)", 
    required=False)
arg_parser.add_argument("--detector", default="homo", type=str, required=False,
    help = "the name of the detector (and therefor the name of the hit tree in the input file)")
arg_parser.add_argument("--max-n-events", "-n", default=None, type=int, required=False,
    help = "If set, will only process the first <n> events")

## options related to the construction of the hits
arg_parser.add_argument("--normalise", action='store_true', required=False)
arg_parser.add_argument("--norm-window-size", type=int, default=3, required=False)
arg_parser.add_argument("--find-2d-peaks", action='store_true', required=False)
arg_parser.add_argument("--allow-2-fiber-hits", action='store_true', required=False)
arg_parser.add_argument("--min-2d-hit-weight", default=0.0, type=float, required=False)
arg_parser.add_argument("--fit-algorithm", default=None, type=str, required=False)

## options related to plotting 
arg_parser.add_argument("--make-gifs", action='store_true', required=False)
arg_parser.add_argument("--make-fiber-hit-plots", action='store_true', required=False)
arg_parser.add_argument("--make-3d-hit-plots", action='store_true', required=False)
arg_parser.add_argument("--charge-cutoff", default=100.0, type=float, required=False)
arg_parser.add_argument("--gif-timesteps", default=1.0, type=float, required=False)

# args to define geometry
arg_parser.add_argument("--homo-centre-x", default=0.0, type=float, required=False)
arg_parser.add_argument("--homo-centre-y", default=30.0, type=float, required=False)
arg_parser.add_argument("--homo-centre-z", default=910.0, type=float, required=False)

arg_parser.add_argument("--plot-centre-x", default=0.0, type=float, required=False)
arg_parser.add_argument("--plot-centre-y", default=30.0, type=float, required=False)
arg_parser.add_argument("--plot-centre-z", default=910.0, type=float, required=False)

arg_parser.add_argument("--plot-size-x", default=2400.0, type=float, required=False)
arg_parser.add_argument("--plot-size-y", default=2400.0, type=float, required=False)
arg_parser.add_argument("--plot-size-z", default=1300.0, type=float, required=False)

args = arg_parser.parse_args(sys.argv[1:])

# some checks on validity of arguments
if args.norm_window_size %2 == 0:
    raise ValueError("norm-window-size must be odd")

def main():
    processor = EventProcessor(args, args.input_file, args.make_fiber_hit_plots, args.make_3d_hit_plots)

    if(args.use_true_hits):
        processor.read_hit_info(f"{args.detector}_truthhits")
        if not args.no_truth_scaling:
            processor.apply_weight_scaling()

    elif (args.use_raw_hits):
        processor.read_hit_info(f"{args.detector}_rawhits")

    else:
        processor.read_hit_info(f"{args.detector}hits")

    processor.process()

    processor.finalise()


if __name__ == "__main__":
    main()

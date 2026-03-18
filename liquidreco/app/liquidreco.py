"""LiquidReco main app.

Takes a root file that contains 2D fiber hits, creates an EventProcessor
based on the provided arguments, and processes the fiber hits using this.

:raises ValueError: if input args are not valid
"""

import sys

from liquidreco.configuration import Configuration
from liquidreco.event_processor import EventProcessor

def main():

    ## read command line configuration
    config = Configuration()
    config.parse_args(sys.argv[1:])

    print(f"running modules {[m.__class__.__name__ for m in config.modules]}")

    ## create the processor
    processor = EventProcessor(
        file_name = config.base_args.input_file,
        tree_name = config.base_args.input_tree_name,
        modules = config.modules,
        max_n_events = config.base_args.n_events
    )

    ## run!!!!!
    processor.event_loop()

if __name__ == "__main__":
    main()

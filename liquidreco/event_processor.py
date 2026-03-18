import typing

import numpy as np
import uproot
from tqdm import tqdm

from liquidreco.modules.module_base import ModuleBase
from liquidreco.event import Event

class EventProcessor:
    """Main application loop for liquidreco

    This is responsible for loading up events from the input file and
    looping over them, applying fit modules specified by the user.
    """

    def __init__(
            self, 
            file_name: str, 
            tree_name: str, 
            modules: typing.List[ModuleBase],
            max_n_events: int = None
        ):
        """Initialiser

        :param file_name: Name of the root file containing hit info
        :type file_name: str
        :param tree_name: The name of the tree within the input file that holds the hits
        :type tree_name: str
        :param modules: List of modules to apply to the events
        :type modules: typing.List[ModuleBase]
        :param max_n_events: Only process this many events, defaults to None
        :type max_n_events: int, optional
        """

        self._file_name = file_name
        self._tree_name = tree_name
        self._modules = modules
        self._max_n_events = max_n_events

        self._read_hit_info()

    def _read_hit_info(self) -> None:
        """load up the input root file and read event info from it
        """

        with uproot.open(self._file_name) as input_file:

            ## get info from the hit tree
            hit_tree = input_file[self._tree_name]

            assert hit_tree is not None, f"No hit tree found with the name {self._tree_name}"

            self._positions = np.array(hit_tree["pos"].array(library="np").tolist())
            self._weights   = np.array(hit_tree["charge"].array(library="np").tolist())
            self._times     = np.array(hit_tree["time"].array(library="np").tolist())
            self._event_ids = np.array(hit_tree["event"].array(library="np").tolist())

    def event_loop(self) -> None:
        """Does the loop over all events, applying each specified module to them
        """

        ## loop through each event
        unique_event_ids = np.unique(self._event_ids)

        ## check if max n events was specified, if so only do that many
        event_ids = unique_event_ids
        if self._max_n_events is not None and len(unique_event_ids) > self._max_n_events:
            event_ids = unique_event_ids[:self._max_n_events]

        ## initialise the modules
        for module in self._modules:
            module.initialise()
        
        for event_id in tqdm(event_ids):

            ## positions in the position, weight and time arrays for this event
            event_array_positions = np.where(self._event_ids == event_id)[0]

            ## values for hits belonging to this particular event
            event_positions = self._positions[event_array_positions, ...]
            event_weights   = self._weights[event_array_positions, ...]
            event_times     = self._times[event_array_positions, ...]

            ## set up the event
            event = Event()
            event.add_data("raw_positions", event_positions)
            event.add_data("raw_weights", event_weights)
            event.add_data("raw_times", event_times)
            event.add_data("id", event_id)

            ## run each of the specified modules
            for module in self._modules:
                module.process(event)

        ## tear down the modules
        for module in self._modules:
            module.finalise()
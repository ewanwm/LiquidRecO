from liquidreco.modules.module_base import ModuleBase
from liquidreco.event import Event

from jsonargparse import ArgumentParser

class WeightScaling(ModuleBase):

    def __init__(self):
        super().__init__()

        self.requirements = ["raw_weights"]
        self.outputs = ["raw_weights"]

    def _help(self):
        
        return """Apply scaling to the charges of the events\n
        \n
        Might be useful if you want e.g. to compare truth hits with hits that\n
        have DAQ or fiber effects applied.
        """
    
    def _setup_cli_options(self, parser: ArgumentParser):
        parser.add_argument(
            "--scaling", 
            help="The scaling to apply to the events", 
            required = True, type = float,
        )
    
    def _initialise(self):
        
        self.scaling = self.args.scaling
    
    def _process(self, event: Event) -> None:

        event.add_data("raw_weights", event["raw_weights"] * self.scaling)
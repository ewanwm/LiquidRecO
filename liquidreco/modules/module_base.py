import abc
from abc import ABC
import typing

from jsonargparse import ActionSubCommands, ArgumentParser, Namespace

from liquidreco.event import Event

class ModuleBase(ABC):
    """Base class of modules that can be run over events.
    
    Any requirements should be given by name in the `requirements` member variable.
    Any outputs should be given by name in the `outputs` member variable.
    
    To be overrridden
    -----------------
    
    - _process() - REQUIRED
    - _initialise() - OPTIONAL
    - _finalise() - OPTIONAL
    - _setup_cli_options() - OPTIONAL
    - _help() - OPTIONAL
    
    """

    def __init__(self):

        self.requirements: typing.List[str] = []
        self.outputs: typing.List[str] = []

        self._arg_parser: ArgumentParser = None
        self.args: typing.Union[Namespace, typing.Dict[str, typing.Any]] = None

    @abc.abstractmethod
    def _process(self, event: Event) -> None:
        """This method should be implemented by any modules you write.
        Should process a single event and add any results to the event object.
        
        :param event: event to be processed
        :type event: Event
        :raises NotImplementedError: if not implemented
        """
        
        raise NotImplementedError()
    
    def _initialise(self) -> None:
        """Set up anything here that needs to be initialised
        
        This gets called *AFTER* argument parsing is done so anything that depends on command 
        line inputs needs to go in here and *NOT* in the __init__() method
        """

        pass
    
    def _finalise(self) -> None:
        """Add anything here that is neaded to tear down the object e.g. closing any files, freeing resources etc."""
        pass

    def _setup_cli_options(self, parser: ArgumentParser) -> None:
        """Override this to set up any command line options for this module 

        The parser passed to this will be a subparser associated with this module. you can call any number of 
        commands defined by argparse which will then be available inside of your module via self.args.

        :param parser: The subparser for this module
        :type parser: ArgumentParser
        """

        pass

    def _help(self) -> str:
        """Should return a help string that will be printed to the command line for this module

        Default behaviour is to use the class docstring of the module

        :return: helpful message about your module
        :rtype: str
        """

        return self.__class__.__doc__

    def _check_requirements(self, event: Event) ->None:
        """Checks that the event has all the required inputs for this module

        :param event: the event to check
        :type event: Event
        :raises ValueError: if the event doesn't have the required inputs
        """
        for requirement in self.requirements:
            if not requirement in event.get_keys():
                raise ValueError(f"Event is missing the {requirement} attribute, did you forget to run a previous module?")
            
    def _check_outputs(self, event: Event) ->None:
        """Checks that the event has all the specified outputs

        :param event: the event to check
        :type event: Event
        :raises ValueError: if the event doesn't have the promised outputs
        """
        for output in self.outputs:
            if not output in event.get_keys():
                raise ValueError(f"Event is missing the {output} attribute, did you forget to add it in your code?")

    def process(self, event: Event) -> None:
        """Process a single event

        :param event: event to be processed
        :type event: Event
        :raises ValueError: if the given event does not contain the specified pre-requisite inputs to at the start, or the promised outputs at the end
        """

        ## check that all the required stuff is in the event
        self._check_requirements(event)

        self._process(event)

        ## check that all the promised outputs have been added
        self._check_outputs(event)

    def initialise(self) -> None:
        """Set up the module"""

        self._initialise()

    def finalise(self) -> None:
        """Tear down the module"""

        self._finalise()

    def setup_parser(self, parser: ArgumentParser) -> None:
        """Setup argument parser for this module

        :param parser: The parser to be set up
        :type parser: ArgumentParser
        """

        self._setup_cli_options(parser)
        self._arg_parser = parser

    def parse_args(self, args: typing.List[str]) -> None:
        """Parse arguments for this module passed in from the command line

        This sets the internally used args variable which lets user code access config
        arguments, so this should be called before the module is called on any events.

        :param args: List of arguments as strings
        :type args: typing.List[str]
        :raises ValueError: if the arg parser has not yet been set up for this module
        """

        if self._arg_parser is None:
            raise ValueError("arg parser not set!!! did you forget to call setup_parser()???")
        
        self.args = self._arg_parser.parse_args(args)

    def parse_object(self, cfg_object: typing.Dict[str, typing.Any]) -> None:
        """Parse arguments for this module passed in as a dict

        Use for parsing objects specified via json config.
        This sets the internally used args variable which lets user code access config
        arguments, so this should be called before the module is called on any events.

        :param cfg_object: Dict containing arguments
        :type cfg_object: typing.Dict[str, typing.Any]
        :raises ValueError: if the arg parser has not yet been set up for this module
        """

        if self._arg_parser is None:
            raise ValueError("arg parser not set!!! did you forget to call setup_parser()???")
        
        self.args = self._arg_parser.parse_object(cfg_object)
        
        return self.args

    def help(self) -> str:
        """Get help message for cmd line for this module

        :return: help string
        :rtype: str
        """

        help_str = (
            f"Inputs:  {self.requirements}\n"
            f"Outputs: {self.outputs}\n"
            "----------------------------------------\n"
            f"{self._help()}"
        )

        return help_str

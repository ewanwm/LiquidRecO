import abc
from abc import ABC
import typing

from liquidreco.event import Event

class ModuleBase(ABC):
    """Base class of modules that can be run over events.
    
    Any requirements should be given by name in the `requirements` member variable.
    Any outputs should be given by name in the `outputs` member variable."""

    def __init__(self):

        self.requirements: typing.List[str] = []
        self.outputs: typing.List[str] = []

    @abc.abstractmethod
    def _process(self, event: Event) -> None:
        """This method should be implemented by any modules you write.
        Should process a single event and add any results to the event object.
        
        :param event: event to be processed
        :type event: Event
        :raises NotImplementedError: if not implemented
        """
        
        raise NotImplementedError()
    
    def _finalise(self) -> None:
        """Add anything here that is neaded to tear down the object e.g. closing any files, freeing resources etc."""
        pass

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

    def finalise(self) -> None:
        """Tear down the module"""

        self._finalise()

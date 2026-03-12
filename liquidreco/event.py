import typing

class Event():
    """Holds event data, is read from and written to by the various modules
    """

    def __init__(
        self
    ):
        
        self._data: typing.Dict[str, typing.Any] = {}

    def __getitem__(self, key):

        ## we should have already checked that all the inputs exist in the 
        ## event so don't need to check here, saving some time in event loop

        return self._data[key] 
    
    def get_keys(self) -> typing.List[str]:
        """Get all the data containers that are currently available in this event
        
        :return: List of all the keys in the internal data table
        :rtype: List[str]
        """

        return self._data.keys()
    
    def add_data(self, key: str, data: typing.Any) -> None:
        """Add some information about the event to the internal data table.

        Can be whatever you want, hit positions, total energy deposited, a nickname for the event, go crazy!
        
        :param key: A key for the data being added, used by modules to access the data
        :type key: str
        :param data: The actual data
        :type data: typing.Any
        """

        self._data[key] = data


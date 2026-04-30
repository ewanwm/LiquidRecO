import typing
from liquidreco.base import Singleton

from jsonargparse import ArgumentParser, Namespace

class GeometryManager(metaclass=Singleton):

    def __init__(self):
        
        self._pitches: typing.Dict[str, float] = {"x": None, "y": None, "z": None}
        self._fiber_positions: typing.Dict[str, float] = {"x": None, "y": None, "z": None}
        self._has_fibers: typing.Dict[str, bool] = {"x": None, "y": None, "z": None}
        self._arg_parser = None
        self.args = None

    def setup_cli_options(self, parser: ArgumentParser) -> None:
        """Sets up geometry related command line arguments

        :param parser: The parser to put the arguments into
        :type parser: ArgumentParser
        """

        parser.add_argument(
            "--x-fiber-pitch", 
            help="The X fiber pitch (in mm)", 
            required = False, default = 10.0, type = float,
        )
        parser.add_argument(
            "--y-fiber-pitch", 
            help="The Y fiber pitch (in mm)", 
            required = False, default = 10.0, type = float,
        )
        parser.add_argument(
            "--z-fiber-pitch", 
            help="The Z fiber pitch (in mm)", 
            required = False, default = 10.0, type = float,
        )
        parser.add_argument(
            "--x-fiber-x-pos", 
            help="The x position that x fibers have (in mm)", 
            required = False, default = 0.0, type = float,
        )
        parser.add_argument(
            "--y-fiber-y-pos", 
            help="The y position that y fibers have (in mm)", 
            required = False, default = 30.0, type = float,
        )
        parser.add_argument(
            "--z-fiber-z-pos", 
            help="The z position that z fibers have (in mm)", 
            required = False, default = 910.0, type = float,
        )
        parser.add_argument(
            "--has-x-fibers", 
            help="Whether this geometry includes fibers in the x direction", 
            required = False, default = True, type = bool,
        )
        parser.add_argument(
            "--has-y-fibers", 
            help="Whether this geometry includes fibers in the y direction", 
            required = False, default = True, type = bool,
        )
        parser.add_argument(
            "--has-z-fibers", 
            help="Whether this geometry includes fibers in the z direction", 
            required = False, default = True, type = bool,
        )

    def setup_parser(self, parser: ArgumentParser) -> None:
        """Set the geometry manager to use this argument parser

        :param parser: the parser to use
        :type parser: ArgumentParser
        """

        self.setup_cli_options(parser)
        self._arg_parser = parser


    def parse_args(self, args: typing.List[str]) -> None:
        """Parse arguments for the geometry in from the command line

        This sets the internally used args variable which lets user code access config
        arguments, so this should be called before the geometry manager is used.

        :param args: List of arguments as strings
        :type args: typing.List[str]
        :raises ValueError: if the arg parser has not yet been set up
        """

        if self._arg_parser is None:
            raise ValueError("arg parser not set!!! did you forget to call setup_parser()???")
        
        args = self._arg_parser.parse_args(args)

        self.consume_args(args)

    def parse_object(self, cfg_object: typing.Dict[str, typing.Any]) -> Namespace:
        """Parse arguments for the geometry passed in as a dict

        Use for parsing objects specified via json config.
        This sets the internally used args variable which lets user code access config
        arguments, so this should be called before the geometry manager is used.

        :param cfg_object: Dict containing arguments
        :type cfg_object: typing.Dict[str, typing.Any]
        :raises ValueError: if the arg parser has not yet been set up for this module
        :return: the parsed arguments
        :rtype: Namespace
        """

        if self._arg_parser is None:
            raise ValueError("arg parser not set!!! did you forget to call setup_parser()???")
        
        args = self._arg_parser.parse_object(cfg_object)
        self.consume_args(args) # <- sets self.args
        
        return self.args

    def consume_args(self, args: Namespace) -> None:
        """Consume arguments that have been parsed

        sets the self.args variable

        :param args: the arguments
        :type parser: Namespace
        """

        self.args = args

        self._pitches["x"] = args.x_fiber_pitch
        self._pitches["y"] = args.y_fiber_pitch
        self._pitches["z"] = args.z_fiber_pitch

        self._fiber_positions["x"] = args.x_fiber_x_pos
        self._fiber_positions["y"] = args.y_fiber_y_pos
        self._fiber_positions["z"] = args.z_fiber_z_pos

        self._has_fibers["x"] = args.has_x_fibers
        self._has_fibers["y"] = args.has_y_fibers
        self._has_fibers["z"] = args.has_z_fibers

    def get_pitch(self, fiber_direction:str) -> float:
        """Get the fiber pitch for fibers in a particular direction

        :param fiber_direction: the fiber direction (x, y or z)
        :type fiber_direction: str
        :raises ValueError: if the provided fiber direction is not one of x, y or z
        :return: the pitch for those fibers
        :rtype: float
        """
        
        if not fiber_direction in self._pitches.keys():
            raise ValueError(f"bad fiber direction: {fiber_direction}")
        
        return self._pitches[fiber_direction]
    
    def get_fiber_position(self, fiber_direction:str) -> float:
        """Get the position that defines fibers in a particular direction

        This is used to determine if e.g. a fiber is an "x fiber", which will all have the same x position 
        (generally the central x position of the detector)

        This is because we currently don't have a way of more properly specifying which direction a fiber is. 
        In future we probably should..... 

        :param fiber_direction: the fiber direction (x, y or z)
        :type fiber_direction: str
        :raises ValueError: if the provided fiber direction is not one of x, y or z
        :return: the central position for those fibers
        :rtype: float
        """
        
        if not fiber_direction in self._fiber_positions.keys():
            raise ValueError(f"bad fiber direction: {fiber_direction}")
        
        return self._fiber_positions[fiber_direction]
    
    def x_fiber_x_pos(self) -> float:
        """Get x position of x fiberss

        :return: x position of x fibers
        :rtype: float
        """

        return self._fiber_positions["x"]
    
    def y_fiber_y_pos(self) -> float:
        """Get y position of y fiberss

        :return: y position of y fibers
        :rtype: float
        """

        return self._fiber_positions["y"]
    
    def z_fiber_z_pos(self) -> float:
        """Get z position of z fiberss

        :return: z position of z fibers
        :rtype: float
        """

        return self._fiber_positions["z"]

    def has_fibers(self, fiber_direction:str) -> bool:
        """Check if the geometry has fibers in a particular direction

        :param fiber_direction: the fiber direction (x, y or z)
        :type fiber_direction: str
        :raises ValueError: if the provided fiber direction is not one of x, y or z
        :return: True if the geometry contains fibers in the specified direction
        :rtype: bool
        """
        
        if not fiber_direction in self._fiber_positions.keys():
            raise ValueError(f"bad fiber direction: {fiber_direction}")
        
        return self._has_fibers[fiber_direction]
    
    def has_x_fibers(self) -> bool:
        """Check if geometry contains fibers in the x direction

        :return: True if it does, false if it doesn't
        :rtype: bool
        """

        return self._has_fibers["x"]
    
    def has_y_fibers(self) -> bool:
        """Check if geometry contains fibers in the y direction

        :return: True if it does, false if it doesn't
        :rtype: bool
        """

        return self._has_fibers["y"]
    
    def has_z_fibers(self) -> bool:
        """Check if geometry contains fibers in the z direction

        :return: True if it does, false if it doesn't
        :rtype: bool
        """

        return self._has_fibers["z"]
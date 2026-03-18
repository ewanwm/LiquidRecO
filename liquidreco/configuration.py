import typing

from liquidreco.modules.module_base import ModuleBase
from liquidreco.modules.module_list import ModuleList

from jsonargparse import ArgumentParser, Namespace, ActionConfigFile, dict_to_namespace, namespace_to_dict

from argparse import RawTextHelpFormatter

class Configuration:
    """Class responsible for parsing user configuration

    The parse_args() method is used to parse command line arguments like
    parse_args(sys.argv[1:]). Then specified modules are available in the 
    `modules` attribute and base config arguments like input file, output
    file etc. are available in the `base_args` attribute.
    """

    def __init__(self):
        
        self._commands: typing.List[str] = []

        ## the raw arguments for each module (first entry will be base arguments)
        self._raw_args: typing.List[typing.List[str]] = []

        self._base_parser: ArgumentParser = None

        self._setup_base_parser()

        ## the instances of the modules to run
        self.modules: typing.List[ModuleBase] = []

        ## The "base" arguments 
        self.base_args: Namespace = None

    def _setup_base_parser(self) -> None:
        """Set up the base argument parser - the one that deals with everything that isn't
        some module specific parameter
        """
                
        ## Set up the arg parser instance
        self._base_parser = ArgumentParser(description=__doc__)
        self._base_parser.add_argument(
            "--input-file", "-i", 
            help="Input root file to read raw hit information from", 
            required=True, type=str
        )
        self._base_parser.add_argument(
            "--input-tree-name", "-t", 
            help="The name of the root tree in the input file which has the raw hit info", 
            required=True, type=str
        )
        self._base_parser.add_argument(
            "--output-file", "-o", 
            help="Output file to write reconstruction to [currently not used hehe]", 
            required=False, type=str
        )
        self._base_parser.add_argument(
            "--n-events", "-n", 
            help="The number of events to process. If not specified then all will be processed", 
            default = None, required=False, type=int
        )
        self._base_parser.add_argument(
            "--config", "-c", 
            help="read fit configuration from a config file. If specified, all module commands and arguments will be ignored", 
            default=None, required=False, type=str
        )
        self._base_parser.add_argument(
            "--dump-config", 
            help="Dump the specified command line configuration to a file with the given name and then exit", 
            default=None, required=False, type=str
        )
        
        ## add subparsers for each registered module
        ## these won't actually be used when parsing but allow printing nice unified
        ## help message with all available modules listed
        module_subparsers = self._base_parser.add_subcommands(title = "Modules", required=False)
        for module_class in ModuleList().get_modules():

            ## add parser with the same name as the module
            arg_parser = ArgumentParser(module_class.__name__)
            module_inst = module_class()
            module_subparsers.add_subcommand(module_class.__name__, parser = arg_parser)
            module_inst.setup_parser(arg_parser)
    
    def _parse_cl_args(self, args: typing.List[str]) -> None:
        """Parse command line arguments

        :param args: List of strings from command line (should use sys.argv[1:])
        :type args: typing.List[str]
        :raises ValueError: If a module you are asking for is not recognised
        """

        current_arg_list: typing.List[str] = []

        for arg in args:
            ## if it's an argument for the current command
            if not arg in ModuleList().get_module_names():
                
                current_arg_list.append(arg)

            ## otherwise it must be a command / name of a module
            else:
                
                ## need to make the map entry for the previous command
                self._raw_args.append(current_arg_list)
                current_arg_list = []

                self._commands.append(arg)

        ## add values for the last parsed module
        self._raw_args.append(current_arg_list)


    def _parse_json_config(self, config_file: str) -> None:

        ## TODO
        raise NotImplementedError()
    
    def _dump_to_json(self, file_name: str) -> None:

        ## TODO
        raise NotImplementedError()


    def parse_args(self, args: typing.List[str]) -> None:
        """Parse list of arguments

        You probably want to pass in sys.argv[1:] - this will get you the command line 
        arguments specified by a user

        :param args: The list of arguments to be parsed
        :type args: typing.List[str]
        """

        ## read in the command line
        self._parse_cl_args(args)

        ## parse the base arguments
        self.base_args = self._base_parser.parse_args(self._raw_args[0])

        ## if -c option specified, parse the module options from config file
        if self.base_args.config is not None:

            self._parse_json_config(self.base_args.config)

        if self.base_args.dump_config is not None:

            self._dump_to_json(self)

        ## parse the args for each module and set up the list of modules to be applied
        for module_command, module_args in zip(self._commands, self._raw_args[1:]):

            module_inst = ModuleList().get_module(module_command)()
            arg_parser = ArgumentParser(module_command, description = module_inst.help(), formatter_class=RawTextHelpFormatter)

            module_inst.setup_parser(arg_parser)

            module_inst.parse_args(module_args)

            self.modules.append(module_inst)

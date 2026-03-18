import typing
import json

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

        ## set up the argument parsers
        self._base_parser = None
        self._fit_parser = None
        self._make_config_parser = None
        self._setup_parser()

        ## the instances of the modules to run
        self.modules: typing.List[ModuleBase] = []

        ## the parsed arguments
        self.base_args = None
        self.fit_args = None
        self.make_config_args = None

    def _setup_parser(self) -> None:
        """Set up the base argument parser - the one that deals with everything that isn't
        some module specific parameter
        """
        
        ## Set up the arg parser instance
        self._base_parser = ArgumentParser()
        
        command_subparsers = self._base_parser.add_subcommands(title = "Commands", required=True, dest="command")

        ## set up the sub parsers for each command
        self._setup_fit_parser()
        self._setup_make_config_parser()
    
        ## add the commands
        command_subparsers.add_subcommand("fit", parser = self._fit_parser, help="Do reconstruction on events with a specified list of modules (do liquidreco fit --help to see the list of available modules")
        command_subparsers.add_subcommand("make-config", parser = self._make_config_parser, help="Create a json config file that can be passed to the 'fit' command. Specify the modules to run from the list of available ones. (do liquidreco make-config --help to see them)")

        ## add module list to each subparser for purpose of nice help message
        self._add_module_subparsers(self._fit_parser)
        self._add_module_subparsers(self._make_config_parser)

    def _setup_make_config_parser(self) -> None:
        """Set up the parser that deals with arguments for the "make-config" command

        Will set the _make_config_parser member variable
        """

        ## TODO: add detailed description using description="blabla" argument
        self._make_config_parser = ArgumentParser(
            "make-config")

        self._make_config_parser.add_argument(
            "--file", "-f", 
            help="The file to output the configuration to", 
            required=True, type=str
        )


    def _setup_fit_parser(self) -> None:
        """Set up the parser that deals with arguments for the "fit" command

        Will set the _fit_parser member variable
        """

        ## TODO: add detailed description using description="blabla" argument
        self._fit_parser = ArgumentParser(
            "fit"
        )

        self._fit_parser.add_argument(
            "--input-file", "-i", 
            help="Input root file to read raw hit information from", 
            required=True, type=str
        )
        self._fit_parser.add_argument(
            "--input-tree-name", "-t", 
            help="The name of the root tree in the input file which has the raw hit info", 
            required=True, type=str
        )
        self._fit_parser.add_argument(
            "--output-file", "-o", 
            help="Output file to write reconstruction to [currently not used hehe]", 
            required=False, type=str
        )
        self._fit_parser.add_argument(
            "--n-events", "-n", 
            help="The number of events to process. If not specified then all will be processed", 
            default = None, required=False, type=int
        )
        self._fit_parser.add_argument(
            "--config", "-c", 
            help="read fit configuration from a config file. If specified, all module commands and arguments will be ignored", 
            default=None, required=False, type=str
        )
        

    def _add_module_subparsers(self, parser: ArgumentParser):
        """add subparsers for each registered module

        these won't actually be used when parsing but allow printing nice unified
        help message with all available modules listed

        :param parser: The parser to add modules to
        :type parser: ArgumentParser
        """
        
        subparsers = parser.add_subcommands(title = "Modules", required=False)
        for module_class in ModuleList().get_modules():

            ## add parser with the same name as the module
            arg_parser = ArgumentParser(module_class.__name__)
            module_inst = module_class()
            subparsers.add_subcommand(module_class.__name__, parser = arg_parser)
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
        """Parse a json config file

        This will overwrite the `args` member variable, effectively ignoring any
        previously specified module settings

        :param config_file: The file to read the config from
        :type config_file: str
        """

        ## reset the module list in case user accidentally specified any modules in cmd line
        self.modules = []

        json_config = None

        with open(config_file, "r") as file:
            
            json_config = json.load(file)

        assert "modules" in json_config.keys(), "Did not find 'modules' key in config file!! Is this really a liquidreco config file????"

        ## get the dict defining the config for each module and parse it 
        for module_json in json_config["modules"]:

            module_name = module_json["module"]
            module_config_dict = module_json["config"]

            module_inst = ModuleList().get_module(module_name)()

            arg_parser = ArgumentParser(module_name, description = module_inst.help(), formatter_class=RawTextHelpFormatter)
            module_inst.setup_parser(arg_parser)

            module_inst.parse_object(module_config_dict)

            self.modules.append(module_inst)

    
    def to_json(self) -> typing.Dict[str, typing.Dict[str, typing.Any]]:
        """Dump the module configuration to a json string

        :return: JSON formatted string containing configuration for all specified modules
        :rtype: str
        """

        module_jsons = []

        for module in self.modules:

            ## convert the Namespace to a dict
            module_arg_dict = namespace_to_dict(module.args)
            
            ## build config dict for this module and push back to list
            module_jsons.append(
                {
                    "module": module.__class__.__name__,
                    "config": module_arg_dict
                }
            )

        return {"modules": module_jsons}

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
        if self.base_args.command == "fit":
            self.fit_args = self.base_args["fit"]
        if self.base_args.command == "make-config":
            print(self.base_args)
            self.make_config_args = self.base_args["make-config"]

        ## if -c option specified, parse the module options from config file
        if self.fit_args is not None and self.fit_args.config is not None:

            self._parse_json_config(self.fit_args.config)

        ## parse the args for each module and set up the list of modules to be applied
        for module_command, module_args in zip(self._commands, self._raw_args[1:]):

            module_inst = ModuleList().get_module(module_command)()
            arg_parser = ArgumentParser(module_command, description = module_inst.help(), formatter_class=RawTextHelpFormatter)

            module_inst.setup_parser(arg_parser)

            module_inst.parse_args(module_args)

            self.modules.append(module_inst)

        ## if we are just making config file, dump the json then exit
        if self.make_config_args is not None:

            with open(self.make_config_args.file, "w") as file:
                json.dump(
                    self.to_json(),
                    file,
                    indent=4, sort_keys=True
                )

            exit(0)

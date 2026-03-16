import typing

from liquidreco.modules.module_base import ModuleBase
from liquidreco.modules.peak_finding import HesseRidgeDetection2D, PeakFinder2D
from liquidreco.modules.reconstruction import HoughTransform, LocalMeanDBSCAN
from liquidreco.modules.hit_building import HitBuilder2D, HitBuilder3D

class Singleton(type):
    
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]
    
class ModuleList(metaclass=Singleton):

    def __init__(self):
        
        self._modules: typing.List[typing.Type[ModuleBase]] = []
        self._module_map: typing.List[str, typing.Type[ModuleBase]] = {}
    
    def register(self, module: typing.Type[ModuleBase]) -> None:
        """Register a module

        :param module: The module class to register
        :type module: typing.Type[ModuleBase]
        """

        if module in self._modules:
            print(f"WARNING: Trying to register module {module.__name__} but it has already been registered")
            return
        
        self._modules.append(module)
        self._module_map[module.__name__] = self._modules[-1]

    def get_modules(self) -> typing.List[typing.Type[ModuleBase]]:
        """Get a list of all the registered modules

        :return: All the registered modules
        :rtype: typing.List[typing.Type[ModuleBase]]
        """

        return self._modules
    
    def get_module_names(self) -> typing.List[str]:
        """Get a list of the names of all registered modules

        :return: list of names
        :rtype: typing.List[str]
        """

        return [c.__name__ for c in self._modules]
    
    def get_module(self, name: str) -> typing.Type[ModuleBase]:
        """Get a module by name

        :param name: the name of the module
        :type name: str
        :raises ValueError: If the module that has been asked for has not been registered
        :return: The requested module
        :rtype: typing.Type[ModuleBase]
        """

        if not name in self._module_map.keys():
            raise ValueError(f"Asked for module {name} but it has not been registered!!\nAvailable modules: {self._modules}")
        
        return self._module_map[name]

    
## Register all the modules
ModuleList().register(HesseRidgeDetection2D)
ModuleList().register(PeakFinder2D)
ModuleList().register(HoughTransform)
#ModuleList().register(LocalMeanDBSCAN)
ModuleList().register(HitBuilder2D)
ModuleList().register(HitBuilder3D)

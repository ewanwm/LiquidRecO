
class Singleton(type):
    """Singleton base class
    
    Using this as a base class ensures that there will only ever be one of the derived object.
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

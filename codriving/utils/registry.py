def _register_generic(module_dict, module_name, module):
    assert module_name not in module_dict, print(
        module_name, module_dict, 'defined in several script files')
    module_dict[module_name] = module


class Registry(dict):
    r"""
    A helper class for registering and managing modules

    >>> # build registry    
    >>> REGISTRY_NAME = Registry(name='REGISTRY_NAME')

    >>> # registr function    
    >>> @REGISTRY_NAME.register
    >>> def foo():
    >>>     pasa
    
    >>> # registr class
    >>> @REGISTRY_NAME.register
    >>> class bar():
    >>>     pasa

    >>> # fetch for class construction 
    >>> func = REGISTRY_NAME[module_name](*args, **kwargs)
    >>> # fetch for function call
    >>> obj = REGISTRY_NAME[module_name](*args, **kwargs)
    """
    def __init__(self, name='Registry', **kwargs):
        self.name = name
        super(Registry, self).__init__(**kwargs)

    def register(self, module):
        name = module.__name__
        _register_generic(self, name, module)

        return module

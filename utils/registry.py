"""
类管理器，通过向注册器(使用wrapper)注册类及名称，并在配置中进行配置
"""

class Registry:
    def __init__(self, name):
        self.__obj_map = {}
        self.__name = name

    @property
    def name(self):
        return  self.__name

    def _do_register(self, name: str, obj) -> None:
        assert name not in self._obj_map, f"An object named '{name}' was already registered in '{self._name}' registry!"
        self._obj_map[name] = obj

    def register(self, obj):
        if obj is None:
            def wrapper(func_or_class):
                name = func_or_class.__name__
                self._do_register(name, func_or_class)
                return func_or_class

            return wrapper

            # used as a function call
        name = obj.__name__
        self._do_register(name, obj)

    def get(self, name: str):
        ret = self._obj_map.get(name)
        if ret is None:
            raise KeyError(
                "No object named '{}' found in '{}' registry!".format(name, self._name)
            )
        return ret

    def __contains__(self, name: str) -> bool:
        return name in self._obj_map

    def __repr__(self) -> str:

        return f"Registries: {self._obj_map.items()}"

    def __iter__(self):
        return iter(self._obj_map.items())

        # pyre-fixme[4]: Attribute must be annotated.

    __str__ = __repr__
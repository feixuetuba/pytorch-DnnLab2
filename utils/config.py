"""
配置管理工具
"""
import inspect
import os
import yaml
import functools
import re


def _get_args_from_config(from_config_func, *args, **kwargs):
    """
    Use `from_config` to obtain explicit arguments.

    Returns:
        dict: arguments to be used for cls.__init__
    """
    signature = inspect.signature(from_config_func)
    if list(signature.parameters.keys())[0] != "cfg":
        if inspect.isfunction(from_config_func):
            name = from_config_func.__name__
        else:
            name = f"{from_config_func.__self__}.from_config"
        raise TypeError(f"{name} must take 'cfg' as the first argument!")
    support_var_arg = any(
        param.kind in [param.VAR_POSITIONAL, param.VAR_KEYWORD]
        for param in signature.parameters.values()
    )
    if support_var_arg:  # forward all arguments to from_config, if from_config accepts them
        ret = from_config_func(*args, **kwargs)
    else:
        # forward supported arguments to from_config
        supported_arg_names = set(signature.parameters.keys())
        extra_kwargs = {}
        for name in list(kwargs.keys()):
            if name not in supported_arg_names:
                extra_kwargs[name] = kwargs.pop(name)
        ret = from_config_func(*args, **kwargs)
        # forward the other arguments to __init__
        ret.update(extra_kwargs)

    return ret


def configurable(init_func=None, *, from_config=None):
    if init_func is not None:
        assert (
            inspect.isfunction(init_func)
            and from_config is None
            and init_func.__name__ == "__init__"
        ), "Incorrect use of @configurable. Check API documentation for examples."

        @functools.wraps(init_func)
        def wrapped(self, *args, **kwargs):
            try:
                from_config_func = type(self).from_config
            except AttributeError as e:
                raise AttributeError(
                    "Class with @configurable must have a 'from_config' classmethod."
                ) from e
            if not inspect.ismethod(from_config_func):
                raise TypeError("Class with @configurable must have a 'from_config' classmethod.")

            explicit_args = _get_args_from_config(from_config_func, *args, **kwargs)
            init_func(self, **explicit_args)

        return wrapped

    else:
        if from_config is None:
            return configurable  # @configurable() is made equivalent to @configurable
        assert inspect.isfunction(
            from_config
        ), "from_config argument of configurable must be a function!"

        def wrapper(orig_func):
            @functools.wraps(orig_func)
            def wrapped(*args, **kwargs):
                explicit_args = _get_args_from_config(from_config, *args, **kwargs)
                return orig_func(**explicit_args)

            wrapped.from_config = from_config
            return wrapped

        return wrapper


def load_config_file(yaml_file, **kwargs):
    with open(yaml_file, "r") as fd:
        data = fd.read()
    data = re.sub('"""\b*\n*[\w\W]*\b*\n*"""\b*\n*', '', data)
    data = re.sub("'''\b*\n*[\w\W]*\b*\n*'''\b*\n*", '', data)
    data = data.format(**kwargs)
    cfg = yaml.safe_load(data)
    return cfg


def dump_config_to_file(content:dict, fpath):
    with open(fpath, "w") as fd:
        yaml.dump(content, fd, default_flow_style=False)


class Config:
    def __init__(self, cfg=None, parent=None, name="root", key_prefix="", **kwargs):
        self.__prefix = key_prefix
        self.__current_item = self
        self.parent = parent
        self._keys = []
        self._name = name

        if cfg is not None:
            self.update(cfg, **kwargs)

    def __getitem__(self, item):
        find, value = self.__find_key(item)
        if not find:
            raise KeyError(f"No such key:{item}")
        return value

    def __contains__(self, item):
        key = self.__get_key(item)
        curr = self.__current_item
        if key in curr._keys: return True
        while curr.parent != None:
            curr = curr.parent
            if key in curr._keys:
                return True
        return False

    def __setitem__(self, item, value):
        key = self.__get_key(item)
        if type(value) == dict:
            value = Config(value, parent=self, name=key, key_prefix=self.__prefix)
        if key not in self._keys:
            self._keys.append(key)
        setattr(self, key, value)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.pop()
        return False

    def keys(self):
        curr = self.__current_item
        keys = set(curr._keys)
        while curr.parent != None:
            curr = curr.parent
            keys.update(set(curr._keys))
        return keys

    @property
    def current(self):
        return self.__current_item._name

    def items(self):
        curr = self.__current_item
        view = []
        for key in curr._keys:
            if key in view:
                continue
            view.append(key)
            yield key, getattr(curr, key)
        while curr.parent != None:
            curr = curr.parent
            for key in curr._keys:
                if key in view:
                    continue
                view.append(key)
                yield key, getattr(curr, key)

    def get(self, item, value=None, dtype=None):
        find, v = self.__find_key(item)
        if not find:
            v = value
        if dtype is not None and  v is not None :
            v = dtype(v)
        return v

    def update(self, cfg, **kwargs):
        """
        使用cfg的内容替换当前配置，如果当前配置中不含相同配置项，则进行合并
        :param cfg:
        :param kwargs:
        :return:
        """
        if isinstance(cfg, dict):
            self.__load_from_dict(cfg, **kwargs)
        elif isinstance(cfg, str):
            assert os.path.isfile(cfg), "Pass sting param to Config, but the value is not a yaml file"
            assert cfg.endswith(".yaml"), "Pass sting param to Config, but the value is not a yaml file"
            self.__load_from_yaml(cfg, **kwargs)
        elif isinstance(cfg, Config):
            self.merge(cfg)

    def __find_key(self, item):
        key = self.__get_key(item)
        curr = self.__current_item
        if key in curr._keys:
            return True, getattr(curr, key)
        while curr.parent != None:
            curr = curr.parent
            if key in curr._keys:
                return True, getattr(curr, key)
        return False, None

    def merge(self, cfg):
        for key in cfg.keys():
            key = self.__get_key(key)
            if key in self._keys:
                my_value = getattr(self, key)
                if isinstance(my_value, Config):
                    my_value.merge(cfg[key])
                else:
                    setattr(self, key, cfg[key])
            else:
                value = cfg[key]
                if isinstance(value, dict):
                    setattr(self, key, Config(value))
                else:
                    setattr(self, key, value)
                self._keys.append(key)

    def __load_from_yaml(self, yaml_file, **kwargs):
        cfg = load_config_file(yaml_file, **kwargs)
        self.__load_from_dict(cfg)

    def __load_from_dict(self, cfg:dict, **kwargs):
        if "base_config" in cfg:
            parent_cfg = cfg["base_config"]
            del cfg["base_config"]
            self.__load_from_yaml(parent_cfg, **kwargs)
            # print(self.MODEL.BACKBONE.NAME)
        self.merge(cfg)

    def __get_key(self, name):
        return f"{self.__prefix}{name}"

    def use(self, subItem):
        key = self.__get_key(subItem)
        find, value = self.__find_key(key)
        v_type = type(value)
        assert isinstance(value,  Config), f" {subItem} is not a {v_type}, not Config object"
        if find:
            self.__current_item = value
        else:
            assert False, f"item:{subItem} no found in {self.__current_item._name} "
        return self

    def use_temp(self, cfg):
        """临时将cfg的内容作为当前配置节点，在pop后将销毁"""
        temp_cfg = Config(cfg, parent=self.__current_item, name="temp_node")
        self.__current_item = temp_cfg
        return self

    def pop(self):
        if self.__current_item.parent is not None:
            self.__current_item = self.__current_item.parent
        return self

    def dump(self, child_only=False):
        """导出当前配置，child_only:只导出当前配置项及其子项"""
        ret = {}
        curr = self.__current_item
        recorded = [curr._name]
        while curr is not None:
            for key in curr._keys:
                if key in recorded:
                    continue
                v = getattr(curr, key)
                if isinstance(v, Config):
                    ret[key] = v.dump(child_only=True)
                else:
                    ret[key] = v
            if child_only:
                curr = None
            else:
                curr = curr.parent
        return ret

    def dump_from_root(self, fpath=None):
        curr = self.__current_item
        self.__current_item = self
        info = self.dump(fpath, child_only=True)
        self.__current_item = curr
        return info

    def clone(self, child_only=False, parent=None):
        """克隆当前配置, child_only:只克隆当前配置项及其子项"""
        newcfg = Config(name=self._name, key_prefix=self.__prefix, parent=parent)
        curr = self.__current_item
        recorded = [curr._name]
        while curr is not None:
            for key in curr._keys:
                if key in recorded:
                    continue
                recorded.append(key)
                v = getattr(curr, key)
                if isinstance(v, Config):
                    newcfg[key] = v.clone(child_only=True, parent=newcfg)
                else:
                    newcfg[key] = v
            if child_only:
                curr = None
            else:
                curr = curr.parent
        return newcfg
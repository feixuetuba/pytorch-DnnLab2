"""
实验配置管理工具
"""

import os
import re

import yaml


class Configure:
    def __init__(self, cfg=None, parent=None, name="root", **kwargs):
        self.__prefix = ""
        self.__current_item = self
        self.__keys = []
        if cfg is not None:
            self.update(cfg, **kwargs)
        if parent is None:
            self.__parent = self
        else:
            self.__parent = parent
        self.__name = name

    @property
    def name(self):
        return self.__name

    @property
    def keys(self):
        return self.__keys
    
    @property
    def current_item(self):
        return self.__current_item.__name

    def dump(self):
        ret = {}
        for key in self.__keys:
            value = getattr(self, key)
            if type(value) == Configure:
                ret[key] = value.dump()
            else:
                ret[key] = value
        return ret
    
    def set_prefix(self, prefix):
        """给每个配置属性加上固定前缀"""
        self.__prefix = prefix

    def get(self, item, value=None):
        key = self.__get_key(item)
        find, v = self.__find_key(key)
        if find: return v
        return value

    def __getitem__(self, item):
        key = self.__get_key(item)
        find, value = self.__find_key(key)
        if not find:
            raise KeyError(f"No such key {item}")
        return value

    def __setitem__(self, item, value):
        key = self.__get_key(item)
        if type(value) == dict:
            value = Configure(value, parent=self, name=key)
        setattr(self.__current_item, key, value)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.pop()

    def update(self, cfg, **kwargs):
        if type(cfg) == dict:
            self.__load_from_dict(cfg)
        elif type(cfg) == str:
            assert os.path.isfile(cfg), "Pass sting param to Configure, but the value is not a yaml file"
            assert cfg.endswith(".yaml"), "Pass sting param to Configure, but the value is not a yaml file"
            self.__load_from_yaml(cfg, **kwargs)
        elif type(cfg) == Configure:
            for key in cfg.keys:
                setattr(self, key, cfg[key])
                if key not in self.__keys:
                    self.keys.append([key])

    def use(self, item):
        """
            使用某个子配置作为主配置，即查找属性时优先使用子配置项的配置。在配置使用完后可用pop函数将主配置项替换回父配置项
            item: 配置项额名称.
            由于不是真正的进行值替换，如果此时使用getattr获取key对应的内容读取的仍然时root的配置，这可能会导致一些异常，
            此时只能使用__getitem__ 方法获取具体的值
        """
        key = self.__get_key(item)
        assert key in self.__current_item.keys, f"No such Configure item:{key}"
        self.__current_item = getattr(self.__current_item, key)
        return self

    def pop(self):
        self.__current_item = self.__current_item.__parent
        return self

    def __find_key(self, key):
        curr = self.__current_item
        if key in curr.keys:
            return True, getattr(curr, key)
        while curr.__parent != self:
            curr = curr.__parent
            if key in curr.keys:
                return True, getattr(curr, key)
        return False, None

    def __load_from_yaml(self, yaml_file, **kwargs):
        with open(yaml_file)  as fd:
            cfg = fd.read()
            cfg = re.sub('"""\b*\r*\n*.*\r*\n*"""\b*\r*\n', '', cfg)
            cfg = cfg.format(**kwargs)
            cfg = yaml.safe_load(cfg)
            self.__load_from_dict(cfg)

    def __load_from_dict(self, cfg:dict):
        for key, value in cfg.items():
            key = self.__get_key(key)
            if type(value) == dict:
                setattr(self, key, Configure(value,parent=self, name=key))
            else:
                setattr(self, key, value)
            if key not in self.__keys:
                self.__keys.append(key)

    def __get_key(self, name):
        return f"{self.__prefix}{name}"


if __name__ == "__main__":
    cfg = Configure(cfg="../checkpoints/test_config.yaml", EXP_ROOT="checkpoints/test")
    with open("check.yaml", "w") as fd:
        yaml.dump(cfg.dump(), fd)

    with cfg.use("dataset"):
        print(f"curr item:{cfg.current_item}")
        print("src:", cfg["src"])
        print("root:", cfg["root"])
        with cfg.use("train"):
            print("\tcurr item:",cfg.current_item)
            print("\tsrc:", cfg["src"])
            print("\troot:", cfg["root"])
            with cfg.use("ds1"):
                print("\t\tcurr item:",cfg.current_item)
                print("\t\t\tsrc:", cfg["src"])
                print("\t\t\troot:", cfg["root"])
                print("\t\t\tlist_file:", cfg["list_file"])
                print("\t\t\ttransform:", cfg["transform"])
                print("\t\t\tcls:", cfg["cls"])
                print("\t\t\ttar:", cfg["tar"])

        with cfg.use("test"):
            print("\tcurr item:",cfg.current_item)
            print("\tsrc:", cfg["src"])
            print("\troot:", cfg["root"])
            print("\tlist_file:", cfg.get("list_file", "No such item"))
            print("\ttransform:", cfg.get("transform", "No such item"))
            print("\tcls:", cfg["cls"])
            print("\ttar:", cfg.get("tar", "No such item"))
        print(f"curr item:{cfg.current_item}")
        print("src:", cfg["src"])
        print("root:", cfg["root"])
        print("list_file:", cfg.get("list_file", "No such item"))
        print("transform:", cfg.get("transform", "No such item"))
        print("cls:", cfg["cls"])
    print(f"curr item:{cfg.current_item}")


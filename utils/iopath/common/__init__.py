# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from utils.iopath.common.file_io import LazyPath, PathManager, file_lock, get_cache_dir, PathManagerFactory
g_pathmgr: PathManager = PathManagerFactory.get(defaults_setup=True)

__all__ = [
    "LazyPath",
    "PathManager",
    "get_cache_dir",
    "file_lock",
]

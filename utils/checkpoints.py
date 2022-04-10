import os
import time
from collections import OrderedDict
import logging
import torch
import yaml

from utils.config import load_config_file


def save_ckpt(state, cfg, epoch, **kwargs):
    state_dict = OrderedDict()
    exp_dir = cfg['EXP_DIR']
    save_dir = cfg['save_dir']
    exp_key = cfg["exp_key"]
    name = cfg['net_name']
    state_dict['save_time'] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    state_dict['config'] = cfg.dump_from_root()
    state_dict['state'] = state
    state_dict['epoch'] = epoch

    for k, v in kwargs.items():
        state_dict[k] = v

    record_file = f"{exp_dir}/exp_record.yaml"
    if not os.path.isfile(record_file):
        record = {}
    else:
        record = load_config_file(record_file)
    if record is None:
        record = {}
        print("save_epoch, Recode Object is None ???")
    if exp_key not in record:
        record[exp_key] = {"min":epoch, "max":epoch}
    else:
        record[exp_key]["max"] = epoch
    with open(record_file, "w") as fd:
        yaml.dump(record, fd, default_flow_style = False)

    torch.save(state_dict, f"{save_dir}/{name}_{epoch}.pth",
                    _use_new_zipfile_serialization=False)

def load_ckpt(cfg, epoch, map_location="cpu"):
    exp_dir = cfg['EXP_DIR']
    save_dir = exp_dir
    name = cfg['net_name']
    ckpt_file = f"{save_dir}/{name}_{epoch}.pth"
    if os.path.isfile(ckpt_file):
        logging.info(f"Load {save_dir}/{name}_{epoch}.pth")
        sd = torch.load(f"{save_dir}/{name}_{epoch}.pth", map_location=map_location)
        return sd
    record_file = f"{exp_dir}/exp_record.yaml"
    if os.path.isfile(record_file):
        records = load_config_file(record_file, map_location=map_location)
        for key, record in records.items():
            min_epoch = record['min']
            max_epoch = record['max']
            if min_epoch <= epoch <= max_epoch:
                save_dir = f"{exp_dir}/{key}"
                break
    else:
        raise FileNotFoundError(f"find no chekcpoint for eopoch:{epoch} in experiment:{save_dir}")
    logging.info(f"Load {save_dir}/{name}_{epoch}.pth")
    sd = torch.load(f"{save_dir}/{name}_{epoch}.pth", map_location=map_location)
    return sd
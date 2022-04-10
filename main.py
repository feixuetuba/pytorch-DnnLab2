import os
import shutil
import sys
import time

import torch

from experiments import load_experiment
from argparse import ArgumentParser
import logging
from utils.config import Config, dump_config_to_file
from utils.distributed import is_main_process, launch

logging.basicConfig(level=logging.INFO,format='%(levelname)s %(asctime)s %(filename)s[line:%(lineno)d] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S')

def get_opts():
    parser = ArgumentParser()
    parser.add_argument("cfg", help="path to config file")
    parser.add_argument("operation", default="test", help="solver operation")
    parser.add_argument("--images", help="the input directory for test")
    parser.add_argument("--which_epoch", type=int, default=-1, help="which epoch to load")
    parser.add_argument("--output", type=str, default=None, help="wich directory ot save test results")
    parser.add_argument("--gpus", type=str, default="0,1,2,3,4,5,6,7,8", help="wich directory ot save test results")
    parser.add_argument("--suffix", type=str, default="", help="experiment_suffix")
    parser.add_argument("--batch_size", type=int, default=-1, help="train batch size")
    return parser.parse_args()


def main(opts, cudas):
    cwd = os.getcwd()
    sys.path.insert(0, cwd)
    logging.basicConfig(level=logging.DEBUG)

    local_time = time.localtime()

    full_time = time.strftime("%Y/%m/%d %H:%M:%S", local_time)
    date_only = time.strftime("%Y_%m_%d", local_time)

    default_exp_dir = os.path.dirname(opts.cfg)
    project_dir = os.path.dirname(os.path.abspath(__file__))

    config = Config(opts.cfg,
                    EXP_DIR = default_exp_dir,
                    PROJECT_DIR = project_dir,
                    )

    exp_name = config["exp_name"]
    version = config["version"]
    config['stage'] = opts.operation
    config['test_images'] = opts.images
    config['exp_date'] = full_time
    config['exp_key'] = date_only
    config['cudas'] = cudas

    if opts.suffix != "":
        config['exp_key'] = f"{date_only}_{opts.suffix}"

    n_gpu = len(cudas)
    if opts.batch_size > 0:
        config["dataset"]["train"]["batch_size"] = -1
        if opts.batch_size % n_gpu == 0:
            config["dataset"]["train"]["bs_per_cuda"] = int(opts.batch_size / n_gpu)
        else:
            config["dataset"]["train"]["batch_size"] = opts.batch_size

    if opts.which_epoch >= 0:
        config['which_epoch'] = opts.which_epoch

    if len(config.get("EXP_DIR", "")) == 0:
        config["EXP_DIR"] = default_exp_dir
    else:
        default_exp_dir = config["EXP_DIR"]

    if opts.operation == "train":
        if "save_dir" not in config:
            save_dir = f"{default_exp_dir}/{config['exp_key']}"
            config["save_dir"] = save_dir
        save_dir = config["save_dir"]
        default_log_dir = f"{save_dir}/log"
        config["sample_dir"] = f"{save_dir}/samples"
        config["log_dir"] = default_log_dir
        config["txt_dir"] = f"{save_dir}/txt"
        if is_main_process():
            txt_dir = config["txt_dir"]
            os.makedirs(save_dir, exist_ok=True)
            os.makedirs(config["sample_dir"], exist_ok=True)
            os.makedirs(txt_dir, exist_ok=True)
            os.makedirs(config["log_dir"], exist_ok=True)
            os.makedirs(default_exp_dir, exist_ok=True)
            shutil.copy(opts.cfg, f"{txt_dir}/{os.path.basename(opts.cfg)}")
            cfg_file = f"{save_dir}/txt/exp_dump_{date_only}.yaml"
            dump_config_to_file(config.dump(), cfg_file)

            cmd = "python " + " ".join(sys.argv)
            with open(f"{save_dir}/txt/cmds.txt", "a+") as fd:
                fd.write(f"{full_time}, {cmd}\n")

    else:

        if opts.output is None:
            save_dir = f"test_results/{exp_name}/{version}/{date_only}"
        else:
            save_dir = opts.output
        config['save_dir'] = save_dir
        if is_main_process():
            os.makedirs(save_dir, exist_ok=True)
            with open(f"{save_dir}/exp_info.txt", "w") as fd:
                fd.write(f"DATE:{full_time}\n")
                fd.write(f"inputs:{opts.images}\n")
                fd.write(f"config:{opts.cfg}\n")
                fd.write(f"epoch:{opts.which_epoch}\n")

    if n_gpu > 0:
        config["device"] = torch.device("cuda")
    else:
        config["device"] = torch.device("cpu")

    config['mute'] = not is_main_process()
    experiment = load_experiment(config)
    func = getattr(experiment, opts.operation)
    func()

if __name__ == "__main__":
    opts = get_opts()

    n_gpu = torch.cuda.device_count()
    gpus = []
    for id in opts.gpus.split(","):
        if int(id) < n_gpu:
            gpus.append(id)

    if opts.operation == "train":
        launch(
            main,
            num_gpus_per_machine=n_gpu,
            machine_rank=0,
            dist_url="auto",
            args=(opts, gpus)
        )
    else:
        main(opts, gpus)
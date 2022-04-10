"""
基于皮肤Mask引导的匀肤模型
基于n2nd_Adv修改，不含adv-loss版本
"""
import copy
import cv2
import logging
import os
import shutil

import numpy as np

from tensorboardX import SummaryWriter
from torchvision.utils import save_image
from tqdm import tqdm

import torch

from experiments.yolo.yolo import Model
from utils import distributed
from utils.checkpoints import load_ckpt, save_ckpt
from utils.files import is_img, load_file_list
from utils.distributed import get_rank
from utils.yolo.datasets import letterbox
from utils.yolo.general import non_max_suppression


class GeneralizedModel:
    def __init__(self, cfg):
        self.cfg = cfg.clone()

        self.network = Model(cfg)
        self.network.eval()
        self.device = cfg['device']

        self.__state_dict = None
        if cfg.get('which_epoch', -1) != -1:
            state_dict = load_ckpt(cfg, cfg['which_epoch'])
            logging.info(f"Load epoch:{cfg['which_epoch']}")
            self.network.load_state_dict(state_dict)
        self.network.to(self.device)

    def __call__(self, **inputs):
        return self.test()

    def set_input(self, **kwargs):
        """图片需为RGB模式, """
        assert "image" in kwargs, "No 'images' input"
        assert "reference" in kwargs, "No 'reference' input"
        for key in ["image", "reference", "target", "real", "mask"]:
            if key in kwargs:
                value = kwargs[key]
                vtype = type(value)
                if vtype is list:
                    value = np.ascontiguousarray(value)
                    value = torch.from_numpy(value)
                elif vtype is np.ndarray:
                    value = torch.from_numpy(value)
                elif type(value) == torch.Tensor:
                    assert len(value.shape) == 4, "Tensor should be [batch, channel, h, w]"
                value = value.to(self.device).float()
                value = value.permute((0, 3, 1, 2)) / 127.5 - 1
                setattr(self, key, value.contiguous())

    def train(self):
        cfg = self.cfg
        self.network.train()
        train_loader, ds_info = mk_dataloader(self.cfg, "train")
        logging.info(ds_info)

        which_epoch = max(0, int(cfg.get('which_epoch', 0)))
        save_freq = cfg.get('save_freq', 1)
        n_epoch = int(cfg.get('n_epoch', 200))
        n_epoch_static = cfg.get('n_epoch_static', n_epoch)
        n_epoch_decay = n_epoch - n_epoch_static
        n_iter_show = cfg.get('n_iter_show', len(train_loader) - 1)
        global_iters = len(train_loader) * which_epoch
        total_iters = n_epoch * len(train_loader)

        discriminator = load_network(cfg, "discriminator")
        if self.__state_dict is not None:
            if 'D' in self.__state_dict['state']:
                discriminator.load_state_dict(self.__state_dict['state']['D'])
                del self.__state_dict['state']['D']
        discriminator.to(self.device)

        criterion_percept = lpips.PerceptualLoss(model='net-lin', net='vgg', use_gpu=True, gpu_ids=cfg['cudas'])
        requires_grad(criterion_percept, False)

        criterion_adv = MultiScaleGANLoss().to(self.device)
        criterion_mse = torch.nn.MSELoss().to(self.device)
        lambda_construct = cfg.get('lambda_construct', 1.0)
        lambda_vgg = cfg.get('lambda_vgg', 1.0)
        lambda_adv = cfg.get('lambda_adv', 1.0)
        lambda_mask = cfg.get('lambda_mask', 1.0)
        sample_dir = cfg['sample_dir']
        del self.__state_dict
        self.__state_dict = None
        if distributed.is_main_process():
            # save_dir = cfg['save_dir']
            log_dir = cfg['log_dir']
            if which_epoch <= 0:
                if os.path.isdir(log_dir):
                    shutil.rmtree(log_dir)
            if os.path.isdir(sample_dir):
                shutil.rmtree(sample_dir)
            os.makedirs(log_dir, exist_ok=True)
            os.makedirs(sample_dir, exist_ok=True)
            writer = SummaryWriter(logdir=log_dir)
            pbar = tqdm(total=total_iters // 2, initial=global_iters, desc=cfg['exp_name'])

        lr = float(cfg['lr'])

        optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
        optimizerD = torch.optim.Adam(discriminator.parameters(), lr=lr)

        epoch_remain = n_epoch - max(which_epoch, 0)
        if n_epoch_decay > 0:
            lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lr_lambda=lambda epoch: (epoch_remain - epoch) / n_epoch_decay,
                last_epoch=-1)
        else:
            lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lr_lambda=lambda epoch: 1,
                last_epoch=-1)

        if distributed.is_main_process():
            writer.add_scalar("lr", lr_scheduler.get_last_lr()[0], which_epoch + 1)
        discriminator = torch.nn.parallel.DistributedDataParallel(discriminator,
                                                                  device_ids=[distributed.get_local_rank()],
                                                                  broadcast_buffers=False,
                                                                  )  # find_unused_parameters=True)

        self.network = torch.nn.parallel.DistributedDataParallel(self.network,
                                                                 device_ids=[distributed.get_local_rank()],
                                                                 broadcast_buffers=False,
                                                                 find_unused_parameters=True)

        iter_per_epoch = len(train_loader)
        data_iter = iter(train_loader)
        logging.info(f"Rank{get_rank()} ready!, {iter_per_epoch}/epoch")
        loss_dict = {}
        latest_loss = 0
        for epoch in range(which_epoch + 1, n_epoch + 1):
            for i in range(iter_per_epoch):
                inputs = next(data_iter)
                self.set_input(**inputs)
                pred = self.__inference()
                total_loss = 0

                loss_dict['gan_loss'] = lambda_adv * criterion_adv(pred[:, :3], self.real, discriminator, optimizerD)
                loss_dict['vgg_restoration'] = lambda_vgg * criterion_percept(pred[:, :3], self.target).sum()
                loss_dict['l2_restoration'] = lambda_construct * criterion_mse(pred[:, :3], self.target)

                if lambda_mask > 0 and hasattr(self, "mask") and pred.shape[1] == 4:
                    loss_dict["mask"] = lambda_mask * criterion_mse(pred[:, 3:4], self.mask)

                for value in loss_dict.values():
                    total_loss += value
                if latest_loss != 0 and total_loss - latest_loss > 30:
                    rank = get_rank()
                    samples = []
                    N = min(3, self.image.shape[0])
                    for _i in range(N):
                        samples.extend(
                            [
                                self.image[_i:_i + 1, :3],
                                self.reference[_i:_i + 1, :3],
                                pred[_i:_i + 1, :3],
                                self.target[_i:_i + 1, :3]
                            ])
                        if pred.shape[1] > 3:
                            samples.append(pred[_i:_i + 1, 3:4].repeat(1, 3, 1, 1))
                    nrow = len(samples) // N
                    samples = torch.cat(samples, dim=0)
                    try:
                        save_image(samples, f"{sample_dir}/{rank}_{global_iters}_corr.jpg", nrow=nrow, normalize=True,
                                   value_range=(-1, 1))
                    except:
                        save_image(samples, f"{sample_dir}/{rank}_{global_iters}_corr.jpg", nrow=nrow, normalize=True,
                                   range=(-1, 1))
                latest_loss = total_loss
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                if not distributed.is_main_process():
                    continue

                global_iters += 1

                desc = ", ".join([f"{k}:{v:.5f}" for k, v in loss_dict.items()])
                pbar.set_description(f"{self.cfg['exp_name']}-{self.cfg['version']} [{epoch}/{n_epoch}] {desc}")
                pbar.update(1)
                writer.add_scalars("loss", loss_dict, global_iters)

                if global_iters % n_iter_show == 0:
                    samples = []
                    N = min(3, self.image.shape[0])
                    for _i in range(N):
                        samples.extend(
                            [
                                self.image[_i:_i + 1, :3],
                                self.reference[_i:_i + 1, :3],
                                pred[_i:_i + 1, :3],
                                self.target[_i:_i + 1, :3]
                            ])
                        if pred.shape[1] > 3:
                            samples.append(pred[_i:_i + 1, 3:4].repeat(1, 3, 1, 1))
                    nrow = len(samples) // N
                    samples = torch.cat(samples, dim=0)
                    try:
                        save_image(samples, f"{sample_dir}/{global_iters}.jpg", nrow=nrow, normalize=True,
                                   value_range=(-1, 1))
                    except:
                        save_image(samples, f"{sample_dir}/{global_iters}.jpg", nrow=nrow, normalize=True,
                                   range=(-1, 1))

            if epoch >= n_epoch_static:
                lr_scheduler.step()

            # lambda_mask*=0.951
            if not distributed.is_main_process():
                continue
            writer.add_scalar("lr", lr_scheduler.get_last_lr()[0], epoch + 1)
            if epoch % save_freq == 0:
                save_ckpt({
                    'generator': self.network.module.state_dict(),
                    'D': discriminator.module.state_dict()
                }, cfg, epoch)

    def test(self, image, conf_thres=0.9, iou_thres = 0.6, input_mode="BGR"):
        """
        对单张已经对齐并缩放到input_height × input_width 的图片进行处理
        :param image:
        :param input_mode:
        :return:
        """
        dest_dims = self.cfg.get('dest_dims', [1024])
        input_h = self.cfg['input_height']
        input_w = self.cfg['input_width']

        assert type(image) == np.ndarray, "image should be uint8 np.ndarray"
        if input_mode != "BGR":
            image = cv2.cvtColor(image, getattr(cv2, f"COLOR_{input_mode}2BGR"))
        net_in, ratio, pad = letterbox(image, (input_h, input_w))
        net_in = net_in.astype(float) / 255.0
        net_in = np.transpose(net_in, (2, 0, 1))[None, ...]

        with torch.no_grad():
            out, train_out = self.network(torch.from_numpy(net_in).float().to(self.device))
            out = non_max_suppression(out, conf_thres, iou_thres, labels=[], multi_label=True, agnostic=False)
            pred = out[0]
            left, top = pad
            pred = pred.cpu().numpy()
            pred[:, 0:4] -= [left, top, left, top]
            pred[:, 0:4] /= ratio[0]

        return pred

    def batch_test(self):
        root_dir = self.cfg['test_images']
        save_root = self.cfg['save_dir']

        os.makedirs(save_root, exist_ok=True)
        save_result = f"{save_root}/outputs"
        os.makedirs(save_result, exist_ok=True)

        files = load_file_list(root_dir)
        labels = []
        label_file = self.cfg.get("label", "")
        if os.path.isfile(label_file):
            with open(label_file) as fd:
                labels = [_.strip() for _ in fd.readlines()]

        for f in tqdm(files, total=len(files), desc="Simple Test"):
            if not is_img(f):
                continue
            img = cv2.imread(f)
            if img is None:
                print(f"Load {root_dir}/{f} failed")
                continue
            pred = self.test(img)
            for predn in pred:
                bbox = predn[:4].astype(int)
                conf = predn[4]
                c = int(predn[5])
                x1, y1, x2, y2 = bbox
                img = cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0))
                if len(labels) >0:
                    c = labels[c]
                c = f"{c}:{conf}"
                img = cv2.putText(img, c, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 255))
            file = os.path.basename(f)
            cv2.imwrite(f"{save_result}/{file}", img)

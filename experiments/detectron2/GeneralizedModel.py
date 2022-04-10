import cv2
import torch

from experiments.detectron2 import build_model
from utils.checkpoints import load_ckpt


class GeneralizedModel:
    def __init__(self, cfg):
        self.cfg = cfg.clone()  # cfg can be modified by model
        self.model = build_model(self.cfg)

        self.model.eval()
        which_epoch = self.cfg.get("which_epoch", "")

        if which_epoch != "":
            self.state_dict = load_ckpt(cfg, which_epoch)
            self.model.load(self.state_dict)

    def set_input(self, **kwargs):
        """image should be RGB Mode"""
        setattr(self, "image", kwargs["image"])
        if "target" in kwargs:
            setattr(self, "target", kwargs["target"])
        self.__img_format = kwargs.get("img_format", "BGR")

    def test(self):
        assert self.state_dict is not None
        self.model.eval()
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            # Apply pre-processing to image.
            original_imnage = self.image
            if self.input_format != "BGR": #== "RGB":
                # whether the model expects BGR inputs or RGB
                # original_image = original_image[:, :, ::-1]
                original_image = cv2.cvtColor(original_imnage, getattr(cv2, f"cv2.COLOR_{self.input_format}2BBGR"))
            height, width = original_image.shape[:2]
            min_input_size = self.cfg.INPUT.MIN_SIZE_TEST
            scale = min(height, width) / min_input_size
            nh = int(height / scale + 0.5)
            nw = int(width / scale + 0.5)
            image = cv2.resize(original_image, (nw, nh))
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

            inputs = {"image": image, "height": height, "width": width}
            predictions = self.model([inputs])[0]
            print("XXXX", predictions)
            return predictions

    def __call__(self, original_image):
        """
        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).

        Returns:
            predictions (dict):
                the output of the model for one image only.
                See :doc:`/tutorials/models` for details about the format.
        """
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            # Apply pre-processing to image.
            if self.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
                original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            image = self.aug.get_transform(original_image).apply_image(original_image)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

            inputs = {"image": image, "height": height, "width": width}
            predictions = self.model([inputs])[0]
            return predictions
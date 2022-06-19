import os
from typing import List, Optional

import torch
from torch import nn
from torchvision import utils

# torch.ops.load_library("transforms/erode/build/liberode.so")
# torch.ops.load_library("transforms/dilate/build/libdilate.so")
# torch.ops.load_library("transforms/scale/build/libscale.so")
torch.ops.load_library("transforms/rotate/build/librotate.so")
# torch.ops.load_library("transforms/resize/build/libresize.so")
# torch.ops.load_library("transforms/translate/build/libtranslate.so")


# TODO: add type annotations to this file


class Rotate(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, params, indices):
        if not isinstance(params[0], float) or params[0] < 0 or params[0] > 360:
            print("Rotation parameter should be a float between 0 and 360 degrees.")
            # raise ValueError
        x_array = list(torch.split(x, 1, 1))
        for i, dim in enumerate(x_array):
            if i in indices:
                d_ = torch.squeeze(dim)
                tf = torch.ops.my_ops.rotate(d_, params[0])
                tf = torch.unsqueeze(torch.unsqueeze(tf, 0), 0)
                x_array[i] = tf
        return torch.cat(x_array, 1)


class ManipulationLayer(nn.Module):
    def __init__(self, layer_id):
        super().__init__()
        self.layer_id = layer_id

        # layers
        # self.erode = Erode()
        # self.dilate = Dilate()
        # self.translate = Translate()
        # self.scale = Scale()
        self.rotate = Rotate()
        # self.resize = Resize()
        # self.flip_h = FlipHorizontal()
        # self.flip_v = FlipVertical()
        # self.invert = Invert()
        # self.binary_thresh = BinaryThreshold()
        # self.scalar_multiply = ScalarMultiply()
        # self.ablate = Ablate()

        self.layer_options = {
            # "erode": self.erode,
            # "dilate": self.dilate,
            # "translate": self.translate,
            # "scale": self.scale,
            "rotate": self.rotate,
            # "resize": self.resize,
            # "flip-h": self.flip_h,
            # "flip-v": self.flip_v,
            # "invert": self.invert,
            # "binary-thresh": self.binary_thresh,
            # "scalar-multiply": self.scalar_multiply,
            # "ablate": self.ablate,
        }

    def save_activations(self, input, index, l_min, l_max):
        if self.layer_id >= l_min and self.layer_id <= l_max:
            x_array = list(torch.split(input, 1, 1))
            for i, activation in enumerate(x_array):
                path = "activations/" + str(self.layer_id) + "/" + str(i) + "/"
                if not os.path.exists(path):
                    os.makedirs(path)
                utils.save_image(
                    torch.squeeze(activation),
                    path + str(index).zfill(5) + ".png",
                    nrow=1,
                    normalize=True,
                    range=(-1, 1),
                )

    def forward(self, input_, tranforms_dict_list: Optional[List[dict]] = None):
        out = input_
        tranforms_dict_list = tranforms_dict_list or []
        for transform_dict in tranforms_dict_list:
            if transform_dict["layer_id"] == -1:
                self.save_activations(
                    input_,
                    transform_dict["index"],
                    transform_dict["params"][0],
                    transform_dict["params"][1],
                )
            if transform_dict["layer_id"] == self.layer_id:
                out = self.layer_options[transform_dict["transform_id"]](
                    out, transform_dict["params"], transform_dict["indices"]
                )
        return out

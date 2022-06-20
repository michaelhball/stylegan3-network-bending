import os
from typing import List, Optional

import torch
from torch import nn
from torch.autograd import Variable
from torchvision import utils

torch.ops.load_library("training/transforms/erode/build/liberode.so")
torch.ops.load_library("training/transforms/dilate/build/libdilate.so")
torch.ops.load_library("training/transforms/scale/build/libscale.so")
torch.ops.load_library("training/transforms/rotate/build/librotate.so")
torch.ops.load_library("training/transforms/resize/build/libresize.so")
torch.ops.load_library("training/transforms/translate/build/libtranslate.so")


# TODO: add type annotations to this file


class Ablate(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, params, indicies):
        x_array = list(torch.split(x, 1, 1))
        for i, dim in enumerate(x_array):
            if i in indicies:
                d_ = torch.squeeze(dim)
                tf = d_ * 0
                tf = torch.unsqueeze(torch.unsqueeze(tf, 0), 0)
                x_array[i] = tf
        return torch.cat(x_array, 1)


class BinaryThreshold(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, params, indicies):
        if not isinstance(params[0], float) or params[0] < -1 or params[0] > 1:
            print("Binary threshold parameter should be a float between -1 and 1.")
            # raise ValueError

        x_array = list(torch.split(x, 1, 1))
        for i, dim in enumerate(x_array):
            if i in indicies:
                d_ = torch.squeeze(dim)
                t = Variable(torch.Tensor([params[0]]))
                t = t.to(d_.device)
                tf = (d_ > t).float() * 1
                tf = torch.unsqueeze(torch.unsqueeze(tf, 0), 0)
                x_array[i] = tf
        return torch.cat(x_array, 1)


class Erode(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, params, indicies):
        if not isinstance(params[0], int) or params[0] < 0:
            print("Erosion parameter must be a positive integer")
            # raise ValueError

        x_array = list(torch.split(x, 1, 1))
        for i, dim in enumerate(x_array):
            if i in indicies:
                d_ = torch.squeeze(dim)
                tf = torch.ops.my_ops.erode(d_, params[0])
                tf = torch.unsqueeze(torch.unsqueeze(tf, 0), 0)
                x_array[i] = tf
        return torch.cat(x_array, 1)


class Dilate(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, params, indicies):
        if not isinstance(params[0], int) or params[0] < 0:
            print("Dilation parameter must be a positive integer")
            # raise ValueError

        x_array = list(torch.split(x, 1, 1))
        for i, dim in enumerate(x_array):
            if i in indicies:
                d_ = torch.squeeze(dim)
                tf = torch.ops.my_ops.dilate(d_, params[0])
                tf = torch.unsqueeze(torch.unsqueeze(tf, 0), 0)
                x_array[i] = tf
        return torch.cat(x_array, 1)


class FlipHorizontal(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, params, indicies):
        x_array = list(torch.split(x, 1, 1))
        for i, dim in enumerate(x_array):
            if i in indicies:
                d_ = torch.squeeze(dim)
                tf = d_.flip([1])
                tf = torch.unsqueeze(torch.unsqueeze(tf, 0), 0)
                x_array[i] = tf
        return torch.cat(x_array, 1)


class FlipVertical(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, params, indicies):
        x_array = list(torch.split(x, 1, 1))
        for i, dim in enumerate(x_array):
            if i in indicies:
                d_ = torch.squeeze(dim)
                tf = d_.flip([0])
                tf = torch.unsqueeze(torch.unsqueeze(tf, 0), 0)
                x_array[i] = tf
        return torch.cat(x_array, 1)


class Invert(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, params, indicies):
        x_array = list(torch.split(x, 1, 1))
        for i, dim in enumerate(x_array):
            if i in indicies:
                d_ = torch.squeeze(dim)
                ones = torch.ones(d_.size(), dtype=d_.dtype, layout=d_.layout, device=d_.device)
                tf = ones - d_
                tf = torch.unsqueeze(torch.unsqueeze(tf, 0), 0)
                x_array[i] = tf
        return torch.cat(x_array, 1)


class Resize(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, params, indicies):
        if not isinstance(params[0], float) or not isinstance(params[1], float):
            print("Resize must have two parameters, which should be positive floats.")
            # raise ValueError
        x_array = list(torch.split(x, 1, 1))
        for i, dim in enumerate(x_array):
            d_ = torch.squeeze(dim)
            print(d_.size())
            tf = torch.ops.my_ops.resize(d_, params[0], params[1])
            print(tf.size())
            tf = torch.unsqueeze(torch.unsqueeze(tf, 0), 0)
            x_array[i] = tf
        return torch.cat(x_array, 1)


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


class ScalarMultiply(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, params, indicies):
        if not isinstance(params[0], float):
            print("Scalar multiply parameter should be a float")
            # raise ValueError

        x_array = list(torch.split(x, 1, 1))
        for i, dim in enumerate(x_array):
            if i in indicies:
                d_ = torch.squeeze(dim)
                tf = d_ * params[0]
                tf = torch.unsqueeze(torch.unsqueeze(tf, 0), 0)
                x_array[i] = tf
        return torch.cat(x_array, 1)


class Scale(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, params, indicies):
        if not isinstance(params[0], float):
            print("Scale parameter should be a float.")
            # raise ValueError
        x_array = list(torch.split(x, 1, 1))
        for i, dim in enumerate(x_array):
            if i in indicies:
                d_ = torch.squeeze(dim)
                tf = torch.ops.my_ops.scale(d_, params[0])
                tf = torch.unsqueeze(torch.unsqueeze(tf, 0), 0)
                x_array[i] = tf
        return torch.cat(x_array, 1)


class Translate(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, params, indicies):
        if (
            not isinstance(params[0], float)
            or not isinstance(params[1], float)
            or params[0] < -1
            or params[0] > 1
            or params[1] < -1
            or params[1] > 1
        ):
            print("Translation must have two parameters, which should be floats between -1 and 1.")
            # raise ValueError
        x_array = list(torch.split(x, 1, 1))
        for i, dim in enumerate(x_array):
            if i in indicies:
                d_ = torch.squeeze(dim)
                tf = torch.ops.my_ops.translate(d_, params[0], params[1])
                tf = torch.unsqueeze(torch.unsqueeze(tf, 0), 0)
                x_array[i] = tf
        return torch.cat(x_array, 1)


class ManipulationLayer(nn.Module):
    def __init__(self, layer_id):
        super().__init__()
        self.layer_id = layer_id

        # layers
        self.ablate = Ablate()
        self.binary_thresh = BinaryThreshold()
        self.erode = Erode()
        self.dilate = Dilate()
        self.flip_h = FlipHorizontal()
        self.flip_v = FlipVertical()
        self.invert = Invert()
        self.resize = Resize()
        self.rotate = Rotate()
        self.scalar_multiply = ScalarMultiply()
        self.scale = Scale()
        self.translate = Translate()

        self.layer_options = {
            "ablate": self.ablate,
            "binary-thresh": self.binary_thresh,
            "erode": self.erode,
            "dilate": self.dilate,
            "flip-h": self.flip_h,
            "flip-v": self.flip_v,
            "invert": self.invert,
            "resize": self.resize,
            "rotate": self.rotate,
            "scalar-multiply": self.scalar_multiply,
            "scale": self.scale,
            "translate": self.translate,
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

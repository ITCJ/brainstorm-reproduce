import numpy as np
import lmdb
import torch
import torch.utils.data as data
import data.util as util


class LQ_label_Dataset(data.Dataset):
    """Read LQ images only in the test phase."""

    def __init__(self, opt):
        super(LQ_label_Dataset, self).__init__()
        self.opt = opt
        self.data_type = self.opt["data_type"]
        self.paths_LQ, self.paths_GT = None, None
        self.LQ_env = None  # environment for lmdb

        self.paths_LQ, self.sizes_LQ = util.get_image_paths(
            self.data_type, opt["dataroot_LQ"]
        )
        assert self.paths_LQ, "Error: LQ paths are empty."

        f = open(str(opt["dataroot_label"]))
        a = f.readlines()
        self.label = []
        for i in a:
            self.label.append(i.split("type:")[1])
        f.close()

    def _init_lmdb(self):
        self.LQ_env = lmdb.open(
            self.opt["dataroot_LQ"],
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

    def __getitem__(self, index):
        if self.data_type == "lmdb" and self.LQ_env is None:
            self._init_lmdb()
        LQ_path = None

        # get LQ image
        LQ_path = self.paths_LQ[index]
        resolution = (
            [int(s) for s in self.sizes_LQ[index].split("_")]
            if self.data_type == "lmdb"
            else None
        )
        img_LQ = util.read_img(self.LQ_env, LQ_path, resolution)
        H, W, C = img_LQ.shape

        if self.opt["color"]:  # change color space if necessary
            img_LQ = util.channel_convert(C, self.opt["color"], [img_LQ])[0]

        # BGR to RGB, HWC to CHW, numpy to tensor
        if img_LQ.shape[2] == 3:
            img_LQ = img_LQ[:, :, [2, 1, 0]]
        img_LQ = torch.from_numpy(
            np.ascontiguousarray(np.transpose(img_LQ, (2, 0, 1)))
        ).float()

        label = self.label[index]
        if str(label) == "1\n":
            label = torch.from_numpy(np.array([0])).long()
        if str(label) == "2\n":
            label = torch.from_numpy(np.array([1])).long()
        if str(label) == "3\n":
            label = torch.from_numpy(np.array([2])).long()
        if str(label) == "4\n":
            label = torch.from_numpy(np.array([3])).long()

        return {"LQ": img_LQ, "LQ_path": LQ_path, "label": label}

    def __len__(self):
        return len(self.paths_LQ)

import os
import json
import cv2
import numpy as np

from torch.utils.data import Dataset

class inference2json_dataset(Dataset):
    def __init__(self, jsonfile, num_joints, inputsize, transform=None):
        self.num_joints = num_joints
        self.db = self._get_db(jsonfile)
        self.inputsize = inputsize
        self.transform = transform

    def _get_db(self, jsonfile):
        with open(jsonfile, "r") as load_f:
            load_dict = json.load(load_f)
        # imgages_name = sorted(os.listdir(image_set))
        # folder_path = os.path.join(path,video_folder)

        return load_dict


    def __len__(self,):
        return len(self.db)

    def __getitem__(self, idx):
        meta = []
        imag_brg = cv2.imread(self.db[idx]['image_path'])
        image_rgb = cv2.cvtColor(imag_brg,cv2.COLOR_BGR2RGB)
        bbox = self.db[idx]['bbox']

        cropped_img = image_rgb[bbox[1]:bbox[3], bbox[0]:bbox[2]]

        s_x = self.inputsize[0] / (bbox[3] - bbox[1])
        s_y = self.inputsize[1] / (bbox[2] - bbox[0])
        scale = np.array([s_x, s_y])

        data = cv2.resize(cropped_img, (self.inputsize[1],self.inputsize[0]))
        # cv2.imshow('demo', data)
        # cv2.waitKey(0)
        if self.transform:
            input = self.transform(data)

        return input, scale

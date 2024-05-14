import os
from typing import List, Union

import cv2
import lmdb
import numpy as np
import pyarrow as pa
import torch
from torch.utils.data import Dataset
from PIL import Image


info = {
    'refcoco': {
        'train': 42404,
        'val': 3811,
        'val-test': 3811,
        'testA': 1975,
        'testB': 1810
    },
    'refcoco+': {
        'train': 42278,
        'val': 3805,
        'val-test': 3805,
        'testA': 1975,
        'testB': 1798
    },
    'refcocog_u': {
        'train': 42226,
        'val': 2573,
        'val-test': 2573,
        'test': 5023
    },
    'refcocog_g': {
        'train': 44822,
        'val': 5000,
        'val-test': 5000
    }
}

def loads_pyarrow(buf):
    """
    Args:
        buf: the output of `dumps`.
    """
    return pa.deserialize(buf)


class RefDataset(Dataset):
    def __init__(self, lmdb_dir, mask_dir, dataset, split, mode, input_size,
                 word_length):
        super(RefDataset, self).__init__()
        self.lmdb_dir = lmdb_dir
        self.mask_dir = mask_dir
        self.dataset = dataset
        self.split = split
        self.mode = mode
        self.input_size = (input_size, input_size)
        self.word_length = word_length
        self.mean = torch.tensor([0.48145466, 0.4578275,
                                  0.40821073]).reshape(3, 1, 1)
        self.std = torch.tensor([0.26862954, 0.26130258,
                                 0.27577711]).reshape(3, 1, 1)
        self.length = info[dataset][split]
        self.env = None

    def _init_db(self):
        self.env = lmdb.open(self.lmdb_dir,
                             subdir=os.path.isdir(self.lmdb_dir),
                             readonly=True,
                             lock=False,
                             readahead=False,
                             meminit=False)
        with self.env.begin(write=False) as txn:
            self.length = loads_pyarrow(txn.get(b'__len__'))
            self.keys = loads_pyarrow(txn.get(b'__keys__'))

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # Delay loading LMDB data until after initialization: https://github.com/chainer/chainermn/issues/129
        if self.env is None:
            self._init_db()
        env = self.env
        with env.begin(write=False) as txn:
            byteflow = txn.get(self.keys[index])
        ref = loads_pyarrow(byteflow)
        # img
        ori_img = cv2.imdecode(np.frombuffer(ref['img'], np.uint8),
                               cv2.IMREAD_COLOR)
        img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
        img_size = img.shape[:2]
        # mask
        seg_id = ref['seg_id']
        mask_dir = os.path.join(self.mask_dir, str(seg_id) + '.png')
        # sentences
        idx = np.random.choice(ref['num_sents'])
        sents = ref['sents']
        # transform
        mat, mat_inv = self.getTransformMat(img_size, True)
        resized_img = cv2.warpAffine(
            img,
            mat,
            self.input_size,
            flags=cv2.INTER_CUBIC,
            borderValue=[0.48145466 * 255, 0.4578275 * 255, 0.40821073 * 255])
        # cv2.imwrite(f"./visualize_output2/{sents[idx]}.png", ori_img)
        if self.mode == 'train' or self.mode == 'val' or self.mode=='test':

            # mask transform
            mask = cv2.imdecode(np.frombuffer(ref['mask'], np.uint8),
                                cv2.IMREAD_GRAYSCALE)
            # mask = cv2.warpAffine(mask,
            #                       mat,
            #                       self.input_size,
            #                       flags=cv2.INTER_LINEAR,
            #                       borderValue=0.)
            mask_indices = np.where(mask == 255)  # 마스크가 1인 값의 좌표
            min_y, min_x = np.min(mask_indices, axis=1)
            max_y, max_x = np.max(mask_indices, axis=1)
            cropped_img = img[min_y:max_y, min_x:max_x]

            # cv2.imwrite(f"./visualize_output2/{sents[idx]}_cropped.png", ori_img[min_y:max_y, min_x:max_x])

            mat, _ = self.getTransformMat(cropped_img.shape[:2], True)
            resized_cropped_img = cv2.warpAffine(
                cropped_img,
                mat,
                self.input_size,
                flags=cv2.INTER_CUBIC,
                borderValue=[0.48145466 * 255, 0.4578275 * 255, 0.40821073 * 255])

            # mask = mask / 255.
            # annot = np.zeros(mask.shape)
            # annot[mask == 1] = 1

            # resized_img, mask = self.convert(resized_img, annot)
            resized_img, resized_cropped_img = self.convert2(resized_img,resized_cropped_img)
            if self.mode =="val" or self.mode =="test":
                idx=0
            sent = sents[idx]

            params = {
                'ori_img': ori_img,
                'seg_id': seg_id,
                'mask_dir': mask_dir,
                'mat':mat,
                'inverse': mat_inv,
                'ori_size': np.array(img_size),
                'sents': sents
            }
            if self.mode =="test":
                return resized_img, sent, resized_cropped_img, params
            else:
                return resized_img, sent, resized_cropped_img

        # elif self.mode == 'val':
        #     # sentence -> vector
        #     sent = sents[0]
        #     img = self.convert(img)[0]
        #     params = {
        #         'mask_dir': mask_dir,
        #         'inverse': mat_inv,
        #         'ori_size': np.array(img_size)
        #     }
        #     return img, sent, params

        else:
            # sentence -> vector
            img = self.convert(img)[0]
            params = {
                'ori_img': ori_img,
                'seg_id': seg_id,
                'mask_dir': mask_dir,
                'inverse': mat_inv,
                'ori_size': np.array(img_size),
                'sents': sents
            }
            return img, params

    def getTransformMat(self, img_size, inverse=False):
        ori_h, ori_w = img_size
        inp_h, inp_w = self.input_size
        scale = min(inp_h / ori_h, inp_w / ori_w)
        new_h, new_w = ori_h * scale, ori_w * scale
        bias_x, bias_y = (inp_w - new_w) / 2., (inp_h - new_h) / 2.

        src = np.array([[0, 0], [ori_w, 0], [0, ori_h]], np.float32)
        dst = np.array([[bias_x, bias_y], [new_w + bias_x, bias_y],
                        [bias_x, new_h + bias_y]], np.float32)

        mat = cv2.getAffineTransform(src, dst)
        if inverse:
            mat_inv = cv2.getAffineTransform(dst, src)
            return mat, mat_inv
        return mat, None

    def convert(self, img, mask=None):
        # Image ToTensor & Normalize
        img = torch.from_numpy(img.transpose((2, 0, 1)))
        if not isinstance(img, torch.FloatTensor):
            img = img.float()
        img.div_(255.).sub_(self.mean).div_(self.std)
        # Mask ToTensor
        if mask is not None:
            mask = torch.from_numpy(mask)
            # if not isinstance(mask, torch.FloatTensor):
            #     mask = mask.float()
        return img, mask

    def convert2(self, img, cropped_img):
        # Image ToTensor & Normalize
        img = torch.from_numpy(img.transpose((2, 0, 1)))
        if not isinstance(img, torch.FloatTensor):
            img = img.float()
        img.div_(255.).sub_(self.mean).div_(self.std)

        cropped_img = torch.from_numpy(cropped_img.transpose((2, 0, 1)))
        if not isinstance(cropped_img, torch.FloatTensor):
            cropped_img = cropped_img.float()
        cropped_img.div_(255.).sub_(self.mean).div_(self.std)

        return img, cropped_img

    def __repr__(self):
        return self.__class__.__name__ + "(" + \
            f"db_path={self.lmdb_dir}, " + \
            f"dataset={self.dataset}, " + \
            f"split={self.split}, " + \
            f"mode={self.mode}, " + \
            f"input_size={self.input_size}, " + \
            f"word_length={self.word_length}"

    # def get_length(self):
    #     return self.length

    # def get_sample(self, idx):
    #     return self.__getitem__(idx)

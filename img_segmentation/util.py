import os
import numpy as np

import torch
import torch.nn as nn

## 네트워크 저장하기
def save(ckpt_dir, net, optim, epoch):
    # ckpt_dir: str dir
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    torch.save({'net': net.state_dict(), 'optim': optim.state_dict()},
               "%s/model_epoch%d.pth" % (ckpt_dir, epoch))

## 네트워크 불러오기
def load(ckpt_dir, net, optim, cuda_idx=None):
    if not os.path.exists(ckpt_dir):
        # ckpt 없다면 기본 셋팅 return
        epoch = 0
        return net, optim, epoch

    # ckpt있다면 학습 마지막 모델 load
    ckpt_lst = os.listdir(ckpt_dir)
    ckpt_lst.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    if cuda_idx == None:
        dict_model = torch.load('%s/%s' % (ckpt_dir, ckpt_lst[-1]))
    else:
        dict_model = torch.load('%s/%s' % (ckpt_dir, ckpt_lst[-1]), map_location=f'cuda:{cuda_idx}')

    net.load_state_dict(dict_model['net'])
    optim.load_state_dict(dict_model['optim'])
    epoch = int(ckpt_lst[-1].split('epoch')[1].split('.pth')[0])

    return net, optim, epoch

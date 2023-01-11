"""
Dataset과 transform 클래스 정의
"""
import os
import numpy as np

import torch
import torch.nn as nn

from torchvision import transforms

## 데이터 로더를 구현하기
class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None, test=False):
        self.data_dir = data_dir
        self.transform = transform
        self.test = test

        # train, val, test dir에 input과 label이 npy로 각각 저장되어있음.
        lst_data = os.listdir(self.data_dir)        # 모든 파일 리스트 가져오기

        lst_label = [f for f in lst_data if f.startswith('label')]
        lst_input = [f for f in lst_data if f.startswith('input')]

        lst_label.sort()
        lst_input.sort()

        self.lst_label = lst_label
        self.lst_input = lst_input

    def __len__(self):
        """_summary_
        얼마나 for를 돌지 정의.
        """
        return len(self.lst_label)

    def __getitem__(self, index):
        """_summary_
        실제 실행은 index에 해당하는 각 파일 한개씩을 처리하게 됨.
        """
        label = np.load(os.path.join(self.data_dir, self.lst_label[index]))
        input = np.load(os.path.join(self.data_dir, self.lst_input[index]))
        f_name = self.lst_input[index]

        # normalization
        # 보통은 바로 normal transform을 적용하는데 미리 이걸 나눌 필요가 있는지는 의문.
        label = label/255.0
        input = input/255.0

        # 3채널이여야함(gray, color 모두 해당)
        if label.ndim == 2:
            label = label[:, :, np.newaxis]
        if input.ndim == 2:
            input = input[:, :, np.newaxis]

        # dict 형태로 내보낸다.
        data = {'input': input, 'label': label}
        

        if self.transform:
            data = self.transform(data)
            
        if self.test:
            data['fname'] = f_name

        return data


## 트렌스폼 구현하기 (torchvision에 이미 구현되어있긴 함. 공부용)
class ToTensor(object):
    def __call__(self, data):
        label, input = data['label'], data['input']

        # np (H, W, C) -> torch(C, H, W)
        label = label.transpose((2, 0, 1)).astype(np.float32)
        input = input.transpose((2, 0, 1)).astype(np.float32)

        data = {'label': torch.from_numpy(label), 'input': torch.from_numpy(input)}

        return data

class Normalization(object):
    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        label, input = data['label'], data['input']

        input = (input - self.mean) / self.std

        data = {'label': label, 'input': input}

        return data

class RandomFlip(object):
    """_summary_
    좌우 상하 flip
    """
    def __call__(self, data):
        label, input = data['label'], data['input']

        if np.random.rand() > 0.5:    # 0.5의 확률 적용
            # 좌우 flip
            label = np.fliplr(label)
            input = np.fliplr(input)

        if np.random.rand() > 0.5:
            # 위아래 flip
            label = np.flipud(label)
            input = np.flipud(input)

        data = {'label': label, 'input': input}

        return data


if __name__ == "__main__":
    """_summary_
    test 용
    """
    
    is_test = True   # fname 체크하려고 만듬
    activate_transform = False
    
    if activate_transform:
        transform = transforms.Compose(
            [Normalization(mean=0.5, std=0.5), RandomFlip(), ToTensor()]
        )
    else:
        transform = None
    
    if transform:
        dataset_db = Dataset(data_dir="./datasets/train/", transform=transform, test=is_test)
    else:
        dataset_db = Dataset(data_dir="./datasets/train/", test=is_test)
        
    input, label = dataset_db[0]['input'], dataset_db[0]['label']
    
    if is_test:
        f_name = dataset_db[0]['fname']
    print("dtype", type(dataset_db))
    print(f"img shape: {input.shape}")    # transform적용시 C, H, W로 변경됨.
    print(f"label shape: {label.shape}")
    
    
    # 이렇게도 가능
    data = dataset_db.__getitem__(0)
    input, label = data['input'], data['label']
    
    # visualization
    import matplotlib.pyplot as plt
    
    plt.subplot(121)
    plt.title(f_name)
    plt.imshow(input.squeeze()) # 1인 dim을 삭제
    plt.subplot(122)
    plt.imshow(label.squeeze())
    # plt.show()
    if transform:
        print(input.type())    # torch.FloatTensor
        plt.savefig("test_transform.png")
    else:
        plt.savefig("test.png")
        
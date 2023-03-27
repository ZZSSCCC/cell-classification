import os
from torch.utils.data import Dataset
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import random
import numpy as np

def rand_bbox(size, lam):
    W = size[1]
    H = size[2]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2

def imgpath_label(root = ''):
    cls2label = {'AGC':0,'ASC-H':1,'ASC-US':2,'HSIL':1,'LSIL':3,'N':4}
    f = open(root)
    x_l = f.readlines()
    x_list = []
    for p in x_l:
        path = p.split('.')[0].strip()+'.jpg'
        if os.path.exists(path):
            x_list.append(path)
    f.close()
    y_list=[]
    for p in x_list:
        y_list.append(cls2label[p.split('/')[5]])

    return x_list, y_list

class B_Data(Dataset):
    def __init__(self, txt_path, transform, mode='train'):
        # with open(txt_path) as input_file:
            # lines = input_file.readlines()
            # img_name = [line.strip().split(' ')[0] for line in lines]
            # img_label = [int(line.strip().split(' ')[-1]) for line in lines]
            # self.img_name = img_name[:len(img_name)//2]
            # self.img_label = img_label[:len(img_label)//2]
        self.img_name, self.img_label = imgpath_label(txt_path)

        if mode == 'train':
            # label_num_dic = {5: 3116, 12: 6769, 10: 2637, 8: 1417, 2: 343, 14: 59, 9: 37, 13: 13,
            #                  7: 22, 11: 90, 1: 68, 6: 13, 0: 61, 4: 9, 3: 12}
            # label_num_dic_intend = {5: 3116, 12: 6769, 10: 2637, 8: 1417*2, 2: 343*8, 14: 59*40, 9: 37*70, 13: 13*200,
            #                  7: 22*120, 11: 90*30, 1: 68*40, 6: 13*300, 0: 61*40, 4: 9*250, 3: 12*300}
            label_num_dic = {0:580, 1:6636, 2:15888, 3:9609, 4:8574}
            label_num_dic_intend = {0:580+6000, 1:6636, 2:15888, 3:9609, 4:8574}

            self.label_path_dic = {}
            for k in range(len(self.img_label)):
                if self.img_label[k] in self.label_path_dic.keys():
                    self.label_path_dic[self.img_label[k]].append(self.img_name[k])
                else:
                    self.label_path_dic[self.img_label[k]] = [self.img_name[k]]

            for k in label_num_dic.keys():
                app = label_num_dic_intend[k] - label_num_dic[k]
                while app>0:
                    self.img_name.append('0')
                    self.img_label.append(k)
                    app-=1

        self.transform = transform
        self.mode = mode

    def __len__(self):
        return len(self.img_name)

    def __getitem__(self, item):
        img_name = self.img_name[item]
        if img_name == '0' and self.mode=='train':
            label = self.img_label[item]
            names = random.choices(self.label_path_dic[label], k=2)
            img0 = Image.open(names[0]).convert('RGB')
            img0 = self.transform(img0)
            img1 = Image.open(names[1]).convert('RGB')
            img1 = self.transform(img1)
            r = random.random()
            if r <= 0.3:
                img = img0
            else:
                alpha = 1.0
                lam = np.random.beta(alpha, alpha)

                bbx1, bby1, bbx2, bby2 = rand_bbox(img1.size(), lam)
                img = img0
                img[:, bbx1:bbx2, bby1:bby2] = img1[:, bbx1:bbx2, bby1:bby2]

                # img = lam * img0 + (1-lam) * img1
        else:
            label = self.img_label[item]
            img = Image.open(img_name).convert('RGB')
            img = self.transform(img)

        return img, label

class H_B_Data(Dataset):
    def __init__(self, fold, transform):

        root = ''
        self.image = []
        self.label = []
        for f in fold:
            root_path = root + f
            for file in os.listdir(root_path):
                self.image.append(os.path.join(root_path,file))
                self.label.append(int(file.split('_')[0]))

        self.transform = transform

    def __len__(self):
        return len(self.image)

    def __getitem__(self, item):
        img_path = self.image[item]
        label = self.label[item]
        img = Image.open(img_path).convert('RGB')

        img = self.transform(img)

        return img, label, img_path

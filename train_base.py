# ---------------------------------data preparation---------------------------------
# import os
# import mmcv
# import shutil
# from skimage import io
# mmcv.mkdir_or_exist('./data/train')
# mmcv.mkdir_or_exist('./data/test')
#
# for phase in ['train', 'test']:
#     with open(f'./data/{phase}.txt') as f:
#         for line in f.readlines():
#             path = '/'.join(line.split()[0].split('/')[-2:])
#             mmcv.mkdir_or_exist(f'./data/{phase}/{path.split("/")[0]}')
#             shutil.copy(f'./data/{path}', f'./data/{phase}/{path}')
# ---------------------------------data preparation---------------------------------
import numpy as np
import torch, sys
from tqdm import tqdm
import torchvision
import torch.nn.functional as F
import torchvision.transforms as T
from resnet import resnet50

model = resnet50(pretrained=True).cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9, weight_decay=1e-4, nesterov=True)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

tf_train = T.Compose([
    T.Resize((224, 224)),
    T.RandomHorizontalFlip(p=0.5),
    T.ToTensor(),
    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

tf_test = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

import torch.nn as nn


class LDAMLoss(nn.Module):

    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30):
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = weight

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)

        index_float = index.type(torch.cuda.FloatTensor)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0, 1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m

        output = torch.where(index, x_m, x)
        return F.cross_entropy(self.s * output, target, weight=self.weight)

train_set = torchvision.datasets.ImageFolder(root='/data1/bc/data/train', transform=tf_train)
test_set = torchvision.datasets.ImageFolder(root='/data1/bc/data/test', transform=tf_test)

train_loader = torch.utils.data.DataLoader(train_set,
                                           batch_size=32,
                                           shuffle=True,
                                           num_workers=8,
                                           drop_last=True)

test_loader = torch.utils.data.DataLoader(test_set,
                                          batch_size=32,
                                          shuffle=True,
                                          num_workers=8,
                                          drop_last=False)

# 0.3
cls_num_list = [61, 68, 343, 12, 9, 3116, 13, 22, 1417, 37, 2637, 90, 6769, 13, 59]
criterion = LDAMLoss(cls_num_list=cls_num_list, max_m=0.5, s=1).cuda()

max_f1 = 0
for epoch in range(40):
    model.train()
    iterator = tqdm(train_loader, file=sys.stdout)
    iterator.set_description("Train epoch-%d" % epoch)

    for data_iter_step, (images, labels) in enumerate(iterator):
        images = images.cuda()
        labels = labels.cuda()

        outputs = model(images)
        loss = F.cross_entropy(outputs, labels)
        # loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    scheduler.step()

    model.eval()
    rn = 0
    pd_labels = []
    gt_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.cuda()

            pd_labels += model(images).argmax(-1).cpu().tolist()
            gt_labels += labels.tolist()

    pd_labels = np.array(pd_labels)
    gt_labels = np.array(gt_labels)
    p = []
    r = []
    for i in range(15):
        rn = ((gt_labels == i) & (pd_labels == i)).sum()
        p.append(rn / ((pd_labels == i).sum() + 1e-6))
        r.append(rn / ((gt_labels == i).sum() + 1e-6))

    mean_p = sum(p) / len(p)
    mean_r = sum(r) / len(r)
    f1 = 2 * mean_p * mean_r / (mean_p + mean_r + 1e-6)
    print(f'Precision: {mean_p}, Recall: {mean_r}, F1: {f1}', )

    if max_f1 < f1:
        max_f1 = f1
        torch.save({'f1': max_f1, 'model': model.state_dict()}, 'base.pth')

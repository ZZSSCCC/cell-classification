import torch
# from utils.lr_scheduler import WarmupMultiStepLR
from utils import create_logger, AverageMeter, accuracy
import os
import shutil
from tensorboardX import SummaryWriter
import time
from torch.utils.data import DataLoader
import argparse
import warnings
import torch.backends.cudnn as cudnn
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from tqdm import tqdm
from torch.optim.lr_scheduler import LambdaLR
import math
from sklearn.metrics import precision_score, recall_score, f1_score
from model import get_model
from data_utils import get_loader
from resnet import resnet50

class WarmupCosineSchedule(LambdaLR):
    def __init__(self, optimizer, warmup_steps, t_total, cycles=.5, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.cycles = cycles
        super(WarmupCosineSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        # progress after warmup
        progress = float(step - self.warmup_steps) / float(max(1, self.t_total - self.warmup_steps))
        return max(0.0, 0.5 * (1. + math.cos(math.pi * float(self.cycles) * 2.0 * progress)))

def train_model(device, trainLoader, model, epoch, epoch_number, optimizer, criterion, logger):

    model.train()
    start_time = time.time()
    number_batch = len(trainLoader)

    all_loss = AverageMeter()
    loss_c = AverageMeter()
    loss_m = AverageMeter()
    acc = AverageMeter()
    epoch_iterator = tqdm(trainLoader, desc='training')
    for i, (image, labels) in enumerate(epoch_iterator):
        cnt = image.shape[0]
        image, labels = image.to(device), labels.to(device)

        output,_,loss_cons = model(image,labels)

        loss_ce = criterion(output, labels)

        loss = loss_ce + loss_cons

        now_result = torch.argmax(torch.nn.Softmax(dim=1)(output), 1)
        now_acc = accuracy(now_result.cpu(), labels.cpu())

        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        all_loss.update(loss.data.item(), cnt)
        loss_c.update(loss_cons.data.item(), cnt)
        loss_m.update(loss_ce.data.item(), cnt)
        acc.update(now_acc, cnt)

        if (i+1) % args.show_step == 0:

            pbar_str = "Epoch:{:>3d}  Batch:{:>3d}/{}  Batch_Loss:{:>5.3f}  Batch_Accuracy:{:>5.2f}%  Batch_Loss_Cons:{:>5.3f} Batch_Loss_ce:{:>5.3f} ".format(
                epoch, i, number_batch, all_loss.val, acc.val * 100, loss_c.val, loss_m.val
            )
            logger.info(pbar_str)
    end_time = time.time()
    pbar_str = "---Epoch:{:>3d}/{}   Avg_Loss:{:>5.3f}   Avg_Loss_c:{:>5.3f}   Epoch_Accuracy:{:>5.2f}%   Epoch_Time:{:>5.2f}min---".format(
        epoch, epoch_number, all_loss.avg, all_loss.avg,  acc.avg * 100, (end_time - start_time) / 60
    )
    logger.info(pbar_str)
    return acc.avg, all_loss.avg


def valid_model(dataLoader, epoch_number, model, criterion, logger, device):
    model.eval()
    # num_classes = dataLoader.dataset.get_num_classes()
    # fusion_matrix = FusionMatrix(num_classes)

    with torch.no_grad():
        all_loss = AverageMeter()
        acc = AverageMeter()
        pred = []
        label = []
        # epoch_iterator = tqdm(dataLoader, desc='testing')
        for i, (image, labels) in enumerate(dataLoader):

            cnt = image.shape[0]

            image, labels = image.to(device), labels.to(device)
            output,_ = model(image)
            loss = criterion(output, labels)

            now_result = torch.argmax(torch.nn.Softmax(dim=1)(output), 1)
            all_loss.update(loss.data.item(), cnt)
            now_acc = accuracy(now_result.cpu(), labels.cpu())
            acc.update(now_acc, cnt)

            label.extend(labels.cpu().tolist())
            pred.extend(now_result.cpu().tolist())

        P = precision_score(label, pred, average='macro')
        R = recall_score(label, pred, average='macro')
        F = f1_score(label, pred, average='macro')
        # pd_labels = np.array(pred)
        # gt_labels = np.array(label)
        # # p = []
        # r = []
        # f = []
        # for i in range(5):
        #     rn = ((gt_labels == i) & (pd_labels == i)).sum()
        #     pr = rn / ((pd_labels == i).sum() + 1e-6)
        #     re = rn / ((gt_labels == i).sum() + 1e-6)
        #     p.append(pr)
        #     r.append(re)
        #     f.append(2*pr*re/(pr+re+1e-6))
        #
        # logger.info(p)
        # logger.info(r)
        # logger.info(f)
        pbar_str = "------- Valid: Epoch:{:>3d}  Valid_Loss:{:>5.3f}  Valid_Acc:{:>5.2f}  P:{:>5.4f}  R:{:>5.4f}  F:{:>5.4f}-------".format(
            epoch_number, all_loss.avg, acc.avg * 100, P, R, F
        )
        logger.info(pbar_str)
    return F, all_loss.avg

def train(args, logger):

    trainLoader, validLoader = get_loader(args)

    best_result, best_epoch, start_epoch = 0, 0, 1
    device = torch.device("cuda")

    model = get_model(args.num_classes)#resnet50(pretrained=True).cuda()#

    criterion = torch.nn.CrossEntropyLoss()#OnehotLoss()#
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [30],  gamma=0.1)
    # scheduler = WarmupCosineSchedule(optimizer, warmup_steps=500, t_total=10000)
    tensorboard_dir = os.path.join('./output', args.name, "tensorboard")
    shutil.rmtree(tensorboard_dir) if tensorboard_dir is not None and os.path.exists(tensorboard_dir) else \
        print('there is not the same name on tensorboard_dir')
    if tensorboard_dir is not None:
        writer = SummaryWriter(log_dir=tensorboard_dir)
    else:
        writer = None

    model_dir = os.path.join('./output', args.name, "models")
    os.makedirs(model_dir) if not os.path.exists(model_dir) else \
        logger.info("This directory has already existed, Please remember to modify your args.name")

    for epoch in range(start_epoch, args.epoch + 1):
        scheduler.step()
        train_acc, train_loss = train_model(device, trainLoader, model, args.epoch, epoch, optimizer, criterion, logger)
        loss_dict, acc_dict = {"train_loss": train_loss}, {"train_acc": train_acc}

        valid_acc, valid_loss = valid_model(validLoader, epoch, model, criterion, logger, device)
        loss_dict["valid_loss"], acc_dict["valid_acc"] = valid_loss, valid_acc
        if valid_acc > best_result:
            best_result, best_epoch = valid_acc, epoch
            # torch.save(model, os.path.join(model_dir, "best_model.pth"))
            torch.save({'state_dict': model.state_dict(),
                    'epoch': epoch,
                    'best_result': best_result,
                    'best_epoch': best_epoch,
                    'scheduler': scheduler.state_dict(),
                    'optimizer': optimizer.state_dict(),}, os.path.join(model_dir, "best_model.pth"))
        logger.info("--------------Best_Epoch:{:>3d}    Best_Acc:{:>5.2f}%--------------".format(best_epoch, best_result * 100))
        writer.add_scalars("scalar/acc", acc_dict, epoch)
        writer.add_scalars("scalar/loss", loss_dict, epoch)
    writer.close()

def test(args):
    trainLoader, validLoader = get_loader(args)

    num_classes = args.num_classes
    device = torch.device("cuda")

    model = get_model(args.num_classes)
    model.load_state_dict(torch.load('/data2/zhangsc/Projects/miccai2023_blood/output/Mix_clsD_moco/models/best_model.pth')['state_dict'])
    model.eval()

    func = torch.nn.Softmax(dim=1)
    acc = AverageMeter()
    preds, labels, img_pathes = [], [], []
    features_dict = {}
    is_right_dict = {}
    vis_class = [0, 119]
    with torch.no_grad():
        for i, (image, label, onehot_sub_target, image_path, meta) in enumerate(validLoader):
            image, label = image.to(device), label.to(device)

            feature = model(image,label, feature_flag=True)

            output = model(feature,label, classifier_flag=True)
            # loss = criterion(output, label)
            score_result = func(output)

            now_result = torch.argmax(score_result, 1)
            preds.extend(now_result.data.cpu().tolist())
            labels.extend(label.data.cpu().tolist())
            img_pathes.extend(image_path)

            now_acc, cnt = accuracy(args, now_result.cpu(), label.cpu())
            acc.update(now_acc, cnt)
            # s_label = torch.argmax(sub_label,dim=-1)
            for l in range(len(label)):
                if label[l].item() in vis_class:
                    if label[l].item() not in features_dict.keys():
                        features_dict[label[l].item()] = [feature[l].data.cpu().numpy()]
                        is_right_dict[label[l].item()] = [label[l].item() if label[l] == now_result[l] else label[l].item()+4]
                    else:
                        features_dict[label[l].item()].append(feature[l].data.cpu().numpy())
                        is_right_dict[label[l].item()].append(label[l].item() if label[l] == now_result[l] else label[l].item()+4)

    print('test acc:  {}......'.format(acc.avg * 100))

    # import pdb; pdb.set_trace()
    pca_data = np.concatenate((np.array(features_dict[vis_class[0]]),np.array(features_dict[vis_class[1]])))#,np.array(features_dict[17]),np.array(features_dict[18])))
    pca_pred = np.concatenate((np.array(is_right_dict[vis_class[0]]),np.array(is_right_dict[vis_class[1]])))
    pca_label = np.concatenate((np.zeros(len(features_dict[vis_class[0]])),np.ones(len(features_dict[vis_class[1]]))))#,2*np.ones(len(features_dict[17])),3*np.ones(len(features_dict[18]))))
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(pca_data)
    X_97 = np.array([x for i, x in enumerate(X_pca) if pca_pred[i] == vis_class[0]])
    X_99 = np.array([x for i, x in enumerate(X_pca) if pca_pred[i] == vis_class[1]])
    X_97_ = np.array([x for i, x in enumerate(X_pca) if pca_pred[i] == vis_class[0]+4])
    X_99_ = np.array([x for i, x in enumerate(X_pca) if pca_pred[i] == vis_class[0]+4])

    X_97_r = np.array([x for i, x in enumerate(X_pca) if pca_label[i] == 0])
    X_99_r = np.array([x for i, x in enumerate(X_pca) if pca_label[i] == 1])
    plt.figure(figsize=[10, 10])
    plt.scatter(X_97[:, 0], X_97[:, 1],label='97')
    plt.scatter(X_99[:, 0], X_99[:, 1],label='99')
    plt.scatter(X_97_[:, 0], X_97_[:, 1], label='97_w')
    plt.scatter(X_99_[:, 0], X_99_[:, 1], label='99_w')
    plt.legend()
    # tsne = TSNE(n_components=2, learning_rate=200, n_iter=1000).fit_transform(pca_data)
    # plt.figure(figsize=(6.5, 6))
    # plt.scatter(tsne[:, 0], tsne[:, 1], c=pca_pred, cmap='tab10')
    # plt.colorbar()
    plt.savefig("./point_distribution/vis_distri_cons_val_pred.png")

    plt.figure(figsize=[10, 10])
    plt.scatter(X_97_r[:, 0], X_97_r[:, 1], label='97')
    plt.scatter(X_99_r[:, 0], X_99_r[:, 1], label='99')
    plt.legend()
    plt.savefig("./point_distribution/vis_distri_cons_val_label.png")

    # with open("output.txt", "a") as f:
    #     for i in range(len(img_pathes)):
    #         f.write('{} '.format(img_pathes[i]))
    #         f.write('{} '.format(preds[i]))
    #         f.write('{} '.format(labels[i]))
    #         f.write('{}'.format(preds[i]==labels[i]))
    #         f.write('\r\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters

    parser.add_argument("--epoch", default=60, type=int)
    parser.add_argument("--num_classes", default=5, type=int)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--weight_decay", default=0.0005, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--name", default='base', type=str)
    parser.add_argument("--dataset", default='peripheral', type=str)
    parser.add_argument("--backbone", default='resnet50', type=str)
    parser.add_argument("--show_step", default=500, type=int)

    parser.add_argument("--batch_size", default=100, type=int)

    args = parser.parse_args()

    logger, log_file = create_logger(args)
    warnings.filterwarnings("ignore")
    cudnn.benchmark = True

    train(args, logger)
    # test(args)

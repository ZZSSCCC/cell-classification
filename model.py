import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F

def get_model(num_classes):
    model = Network_moco(num_classes=num_classes)
    model = torch.nn.DataParallel(model).cuda()

    return model

class Network(nn.Module):
    def __init__(self, num_classes=1000):
        super(Network, self).__init__()

        complete_backbone = torchvision.models.__dict__['resnet101'](pretrained=True)
        self.encoder = nn.Sequential(*(list(complete_backbone.children())[:-1]))
        self.classifier = nn.Linear(2048, num_classes)
        self.t = 1.7
        self.projection = nn.Linear(2048, 128)

    def forward(self, x, labels=None):
        feas = self.encoder(x)
        feas = feas.view(feas.shape[0],-1)
        output = self.classifier(feas)
        if labels is not None:
            feas_cons = self.projection(feas)
            similarity_matrix = F.cosine_similarity(feas_cons.unsqueeze(1), feas_cons.unsqueeze(0), dim=2)
            labels_reshape = torch.reshape(labels,(x.shape[0],1))
            mask_for_same = ((labels_reshape - labels)==0).long() #B x self.K
            assert mask_for_same.shape == similarity_matrix.shape
            similarity_matrix = torch.exp(similarity_matrix / self.t)
            sim = mask_for_same * similarity_matrix
            no_sim = similarity_matrix - sim
            sim[sim==0] = 1e7
            top_min_10, _ = sim.topk(10, 1, False, True)
            top_max_10, _ = no_sim.topk(10, 1, True, True)
            same_min = top_min_10.mean(dim=1)
            diff_max = top_max_10.mean(dim=1)
            difference = (diff_max - same_min) + 0.3
            loss_class_distance = torch.clamp(difference, min=0).mean()

            # loss_class_distance = self.con_loss(feas_cons, labels)#TripletLoss()(feas_cons, labels)#

            return output, feas, loss_class_distance
        else:
            return output, feas

    def load_model(self, model_path):
        pretrain_dict = torch.load(
            model_path, map_location="cpu" if self.cfg.CPU_MODE else "cuda"
        )
        pretrain_dict = pretrain_dict['state_dict'] if 'state_dict' in pretrain_dict else pretrain_dict
        model_dict = self.state_dict()
        from collections import OrderedDict
        new_dict = OrderedDict()
        for k, v in pretrain_dict.items():
            if k.startswith("module"):
                new_dict[k[7:]] = v
            else:
                new_dict[k] = v
        model_dict.update(new_dict)
        self.load_state_dict(model_dict)
        print("Model has been loaded...")

    def con_loss(self, features, labels):
        B, _ = features.shape
        features = F.normalize(features)
        cos_matrix = features.mm(features.t())
        pos_label_matrix = torch.stack([labels == labels[i] for i in range(B)]).float()
        neg_label_matrix = 1 - pos_label_matrix
        pos_cos_matrix = 1 - cos_matrix
        neg_cos_matrix = cos_matrix - 0.4
        neg_cos_matrix[neg_cos_matrix < 0] = 0
        loss = (pos_cos_matrix * pos_label_matrix).sum() + (neg_cos_matrix * neg_label_matrix).sum()
        loss /= (B * B)
        return loss

class Network_moco(nn.Module):
    def __init__(self, num_classes=1000):
        super(Network_moco, self).__init__()

        complete_backbone = torchvision.models.__dict__['resnet34'](pretrained=True)
        complete_backbone_k = torchvision.models.__dict__['resnet34'](pretrained=True)
        self.encoder_q = nn.Sequential(*(list(complete_backbone.children())[:-1]))
        self.encoder_k = nn.Sequential(*(list(complete_backbone_k.children())[:-1]))
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        self.classifier = nn.Linear(512, num_classes)

        # create the queue
        self.K = 9600
        self.m = 0.999
        self.t = 1.7
        self.register_buffer("queue", torch.randn(512, 9600))  # dim = 512, features num 9600
        self.register_buffer("label_queue", -1 * torch.ones(9600, dtype=torch.long))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    def forward(self, x, labels=None):
        feas = self.encoder_q(x)
        feas = feas.view(feas.shape[0], -1)

        output = self.classifier(feas)

        if labels is not None:

            with torch.no_grad():
                self._momentum_update_key_encoder()
                k = self.encoder_k(x)
                k = k.view(k.shape[0], -1)

            self._dequeue_and_enqueue(k, labels=labels)
            queue_k = self.queue.T.clone().detach()

            similarity_matrix = F.cosine_similarity(feas.unsqueeze(1), queue_k.unsqueeze(0), dim=2)
            queue_labels = self.label_queue.clone()
            labels = torch.reshape(labels,(x.shape[0],1))
            mask_for_same = ((labels - queue_labels)==0).long() #B x self.K
            assert mask_for_same.shape == similarity_matrix.shape
            similarity_matrix = torch.exp(similarity_matrix / self.t)
            sim = mask_for_same * similarity_matrix
            no_sim = similarity_matrix - sim
            sim[sim==0] = 1e7
            top_min_10, _ = sim.topk(10, 1, False, True)
            top_max_10, _ = no_sim.topk(10, 1, True, True)
            same_min = top_min_10.mean(dim=1)
            diff_max = top_max_10.mean(dim=1)
            difference = (diff_max - same_min) + 0.3
            loss_class_distance = torch.clamp(difference, min=0).mean()

            # loss_class_distance = self.con_loss(feas_cons, labels)#TripletLoss()(feas_cons, labels)#

            return output, feas, loss_class_distance
        else:
            return output, feas

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, labels):

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        self.label_queue[ptr:ptr + batch_size] = torch.reshape(labels, (batch_size,))  # labels.reshape(batch_size)
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    def load_model(self, model_path):
        pretrain_dict = torch.load(
            model_path, map_location="cpu" if self.cfg.CPU_MODE else "cuda"
        )
        pretrain_dict = pretrain_dict['state_dict'] if 'state_dict' in pretrain_dict else pretrain_dict
        model_dict = self.state_dict()
        from collections import OrderedDict
        new_dict = OrderedDict()
        for k, v in pretrain_dict.items():
            if k.startswith("module"):
                new_dict[k[7:]] = v
            else:
                new_dict[k] = v
        model_dict.update(new_dict)
        self.load_state_dict(model_dict)
        print("Model has been loaded...")

    def con_loss(self, features, labels):
        B, _ = features.shape
        features = F.normalize(features)
        cos_matrix = features.mm(features.t())
        pos_label_matrix = torch.stack([labels == labels[i] for i in range(B)]).float()
        neg_label_matrix = 1 - pos_label_matrix
        pos_cos_matrix = 1 - cos_matrix
        neg_cos_matrix = cos_matrix - 0.4
        neg_cos_matrix[neg_cos_matrix < 0] = 0
        loss = (pos_cos_matrix * pos_label_matrix).sum() + (neg_cos_matrix * neg_label_matrix).sum()
        loss /= (B * B)
        return loss

class TripletLoss(nn.Module):

    def __init__(self, margin=0.3):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        n = inputs.size(0)
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)

        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        return self.ranking_loss(dist_an, dist_ap, y)
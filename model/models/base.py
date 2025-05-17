from abc import ABC
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.networks import res18, resnet
from model.utils import get_summary_writer, Averager


import torch

def sinkhorn(A, epsilon=0.05, max_iters=3):
    Q = torch.exp(A/epsilon)
    B = Q.shape[0]
    K = Q.shape[1]
    Q /= torch.sum(Q)
    for it in range(max_iters):
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= K

        Q /= torch.sum(Q, dim=1, keepdim=True)
        Q /= B
    Q *= B
    return Q






def record_grad_norm(module):
    def hook(grad):
        norms = torch.norm(grad, dim=1)
        norm = torch.sum(norms)
        module.grad_norm_accumulator.add(norm.item(), norms.size(0))

    return hook


class FewShotModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.construct_encoder(args)
        # self.encoder = HeadWrapper(self.encoder, self.hdim, args)
        self.projector = nn.Sequential(
            nn.Linear(self.hdim, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 256),
            nn.BatchNorm1d(256),
        )
        self.projector_t = nn.Sequential(
            nn.Linear(self.hdim, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 256),
            nn.BatchNorm1d(256),
        )
        self.ep = 0
        self.gep = 0
        self.lep = 0
        self.writer = get_summary_writer()
        self.grad_norm_accumulator = Averager()
        self.emb_norm_accumulator = Averager()
        for param_q, param_t in zip(self.encoder.parameters(), self.encoder_t.parameters()):
            param_t.data.copy_(param_q.data)
            param_t.requires_grad = False
        for param_q, param_t in zip(self.projector.parameters(), self.projector_t.parameters()):
            param_t.data.copy_(param_q.data)
            param_t.requires_grad = False
        self.m = 0.99


        self.predictor = nn.Sequential(
            nn.Linear(256, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 256),

        )
        self.mem_bank_size = 16384
        self.momentum = 0.99
        self.register_buffer('queue', F.normalize(torch.randn(self.mem_bank_size, 256)))
        self.register_buffer('queue_ptr', torch.tensor([0]))
        self.temperature = 0.2
        self.sinkhorn_iter = 3
        self.temperature2 = 1.0
        self.shot_sampling = 'top'
        self.num_shots = 20
        self.moco = 5
        self.meta = 0.5

        self.l1 = nn.L1Loss()

    @torch.no_grad()
    def _dequeue_and_enqueue(self, targets):
        batch_size = targets.shape[0]

        ptr = int(self.queue_ptr)
        assert self.mem_bank_size % batch_size == 0

        # replace the targets at ptr (dequeue and enqueue)
        self.queue[ptr:ptr + batch_size] = targets

        ptr = (ptr + batch_size) % self.mem_bank_size  # move pointer

        self.queue_ptr[0] = ptr
    @torch.no_grad()
    def _momentum_update_target_encoder(self):
        for param_q, param_t in zip(self.encoder.parameters(), self.encoder_t.parameters()):
            param_t.data = param_t.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _momentum_update_target_projector(self):
        for param_q, param_t in zip(self.projector.parameters(), self.projector_t.parameters()):
            param_t.data = param_t.data * self.m + param_q.data * (1. - self.m)

    def construct_encoder(self, args):
        if args.backbone_class == 'ConvNet':
            from model.networks.convnet import convnet
            self.hdim = 1600
            self.encoder = convnet(False)
            self.encoder_t = convnet(False)
        elif args.backbone_class == 'Res12':
            self.hdim = 640
            from model.networks.res12 import ResNet
            params = {}
            if args.dataset in ['CIFAR-FS', 'FC100']:
                params['drop_block'] = False
            self.encoder = ResNet(**params)
            self.encoder_t = ResNet(**params)
        elif args.backbone_class in res18.__all__:
            # from model.networks.res18 import ResNet
            self.encoder = getattr(res18, args.backbone_class)()
            self.hdim = 512 * self.encoder.expansion
        elif args.backbone_class in resnet.__all__:
            self.hdim = 64
            # from model.networks.res18 import ResNet
            self.encoder = getattr(resnet, args.backbone_class)()
        elif args.backbone_class == 'WRN':
            self.hdim = 640
            from model.networks.WRN28 import Wide_ResNet
            self.encoder = Wide_ResNet(28, 10,
                                       0.5)  # we set the dropout=0.5 directly here, it may achieve better results by tunning the dropout
        else:
            raise ValueError('Unrecognized network structure')
        print(f"emb size {self.hdim}")

    def split_instances_normal(self, num_tasks, num_shot, num_query, num_way, num_class=None):
        num_class = num_way if (num_class is None or num_class < num_way) else num_class

        permuted_ids = torch.zeros(num_tasks, num_shot + num_query, num_way).long()
        for i in range(num_tasks):
            # select class indices
            clsmap = torch.randperm(num_class)[:num_way]
            # ger permuted indices
            for j, clsid in enumerate(clsmap):
                permuted_ids[i, :, j].copy_(
                    torch.randperm((num_shot + num_query)) * num_class + clsid
                )

        if torch.cuda.is_available():
            permuted_ids = permuted_ids.cuda()

        support_idx, query_idx = torch.split(permuted_ids, [num_shot, num_query], dim=1)
        return support_idx, query_idx

    def cosin_sim(self, x, y):  # x:[n, d]  y:[m, d]
        x = x.unsqueeze(1)  # [n, 1, d]
        y = y.unsqueeze(0)  # [1, m, d]
        return F.cosine_similarity(x, y, dim=-1)
    def split_instances(self, data):
        args = self.args
        if self.training:
            if args.unsupervised:
                return self.split_instances_normal(args.num_tasks, args.shot,
                                                   args.query, args.way, args.batch_size)
            return self.split_instances_normal(args.num_tasks, args.shot,
                                               args.query, args.way, args.num_classes)
        else:
            return self.split_instances_normal(1, args.eval_shot,
                                               args.eval_query, args.eval_way)

    def M(self, C, u, v, eps):
        "Modified cost for logarithmic updates"
        return (-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / eps
    def SinkhornDistance(self, p1, p2, C, itr=5, eps=0.5):
        u = torch.zeros_like(p1).cuda()
        v = torch.zeros_like(p2).cuda()
        for _ in range(itr):
            u = eps * (torch.log(p1 + 1e-12) - torch.logsumexp(self.M(C, u, v, eps), dim=-1)) + u
            v = eps * (torch.log(p2 + 1e-12) - torch.logsumexp(self.M(C, u, v, eps).transpose(-2, -1), dim=-1)) + v

        pi = torch.exp(self.M(C, u, v, eps))
        return (pi * C).sum((-2, -1)).mean()

    def forward(self, x, get_feature=False, epoch=0, **kwargs):
        if get_feature:
            # get feature with the provided embeddings
            return self.encoder(x)
        else:
            if self.training:
                self._momentum_update_target_encoder()
                self._momentum_update_target_projector()
                x1, x2 = x
                z1 = F.normalize(self.predictor(self.projector(self.encoder(x1))), p=2, dim=-1)
                z2 = F.normalize(self.projector(self.encoder(x2)), p=2, dim=-1).detach()
                with torch.no_grad():
                    z2 = F.normalize(self.projector_t(self.encoder_t(x2)), p=2, dim=-1).detach()
                    sim = torch.mm(z2, self.queue.T).detach()
                    similarity, index = torch.topk(sim, k=self.num_shots, dim=1)

                    similarity = F.softmax(similarity, dim=1)
                    similarity = similarity.unsqueeze(dim=2)
                    similarity = similarity.repeat(1, 1, 256)
                    samples = index.reshape(-1)
                    shots = self.queue[samples].clone().detach()
                    prototypes = shots.reshape(sim.shape[0], self.num_shots, -1).mean(dim=1)
                    shots = shots.reshape(sim.shape[0], self.num_shots, -1)
                    proto_new = similarity * shots
                    proto_new = torch.sum(proto_new, dim=1)
                    cluster_sim = self.cosin_sim(F.normalize(proto_new), F.normalize(proto_new))
                    cluster_sim = F.softmax(cluster_sim, dim=1)

                proto_new = F.normalize(proto_new, p=2, dim=-1)
                logit_neighbor = torch.einsum('bc,kc->bk', [z1, proto_new])
                label_neighbor = torch.arange(logit_neighbor.size(0), dtype=torch.long, device=logit_neighbor.device)
                logit_self = torch.einsum('bc, kc->bk', [z1, z2])
                z2_hat = F.normalize(z2 + proto_new, p=2, dim=-1)
                logit_hat = torch.einsum('bc, kc->bk', [z1, z2_hat])


                cost = 1 - self.cosin_sim(z2, z2)
                cost = cost.detach()
                cost = (cost - cost.min(-1, keepdims=True)[0]) / (
                        cost.max(-1, keepdims=True)[0] - cost.min(-1, keepdims=True)[0])
                cost = (8 * cost + torch.eye(sim.size(0)).cuda()).unsqueeze(0).repeat(sim.size(0), 1, 1)



                wcp_1 = self.SinkhornDistance(F.softmax(logit_self, dim=1), cluster_sim, cost)
                wcp_2 = self.SinkhornDistance(F.softmax(logit_neighbor, dim=1), cluster_sim, cost)
                wcp_loss = (wcp_1 + wcp_2) / 2
                loss_fsl = F.cross_entropy(logit_self, label_neighbor)

                kd_pq = F.kl_div(F.log_softmax(logit_self, dim=1), F.softmax(logit_neighbor.detach(), dim=1),
                                 reduction='batchmean')
                kd_qp = F.kl_div(F.log_softmax(logit_neighbor, dim=1), F.softmax(logit_self.detach(), dim=1),
                                 reduction='batchmean')
                loss_n = (kd_qp + kd_pq) / 2

                l_pos = torch.mul(z1, z2).sum(dim=1, keepdim=True)

                l_neg = torch.mm(z1, self.queue.clone().T.detach())
                logits = torch.cat([l_pos, l_neg], dim=1).div(self.temperature)
                labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)

                loss = self.moco * F.cross_entropy(logits, labels) + wcp_loss + loss_n + self.meta * loss_fsl

                self._dequeue_and_enqueue(z2)

                return loss
            else:
                support_idx, query_idx = self.split_instances(x)
                p1 = self.projector(self.encoder(x))
                # p1 = self.encoder(x)
                logits = self._forward(p1, support_idx, query_idx, **kwargs)
                return logits

         
    def _forward(self, instance_embs, support_idx, query_idx, **kwargs):
        # emb_dim = instance_embs.size(-1)

        # organize support/query data
        support = instance_embs[support_idx.flatten()].view(*(support_idx.shape + (-1,)))
        query = instance_embs[query_idx.flatten()]
        query = self.predictor(query)
        query = query.view(*(query_idx.shape + (-1,)))
        return self._forward_task(support, query, **kwargs)

    def _forward_task(self, support, query, **kwargs):
        pass

    def set_epoch(self, ep):
        self.gep = ep
        self.lep = 0
        self.clear_statistics()

    def clear_statistics(self):
        self.grad_norm_accumulator.reset()
        self.emb_norm_accumulator.reset()

    def statistics(self):
        return {'emb_grad_norm': self.grad_norm_accumulator.item(),
                'emb_norm': self.emb_norm_accumulator.item()}

    


class FewShotModelWrapper(FewShotModel, ABC):
    def __init__(self, args, model: FewShotModel):
        super().__init__(args)
        self.model = model
        self.encoder = self.model.encoder

    def load_state_dict(self, state_dict, strict=True):
        self.model.load_state_dict(state_dict, strict)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        return self.model.state_dict(destination, prefix, keep_vars)

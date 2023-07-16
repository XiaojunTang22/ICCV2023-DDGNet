import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import model
import torch.nn.init as torch_init
import random
from edl_loss import EvidenceLoss
from edl_loss import relu_evidence, exp_evidence, softplus_evidence

torch.set_default_tensor_type('torch.cuda.FloatTensor')


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        torch_init.kaiming_uniform_(m.weight)
        if type(m.bias) != type(None):
            m.bias.data.fill_(0)


# baseline (Exp 1 in this paper)
class No_fusion(nn.Module):
    def __init__(self, n_feature, n_class, **args):
        super().__init__()
        self.attentions = nn.ModuleList([nn.Sequential(nn.Conv1d(n_feature, 512, (3,), padding=1),
                                                       nn.LeakyReLU(0.2),
                                                       nn.Dropout(0.5),
                                                       nn.Conv1d(512, 512, (3,), padding=1),
                                                       nn.LeakyReLU(0.2),
                                                       nn.Conv1d(512, 1, (1,)),
                                                       nn.Dropout(0.5),
                                                       nn.Sigmoid()) for _ in range(2)])

    def forward(self, vfeat, ffeat, **args):
        v_atn = self.attentions[0](vfeat)
        f_atn = self.attentions[1](ffeat)
        return v_atn, vfeat, f_atn, ffeat, {}


# DDG_Net without gcn (Exp 2 in this paper)
class DDG_Net_nogcn(nn.Module):
    def __init__(self, n_feature, args):
        super().__init__()

        self.attentions = nn.ModuleList([nn.Sequential(nn.Conv1d(n_feature, 512, (3,), padding=1),
                                                       nn.LeakyReLU(0.2),
                                                       nn.Dropout(0.5),
                                                       nn.Conv1d(512, 512, (3,), padding=1),
                                                       nn.LeakyReLU(0.2),
                                                       nn.Conv1d(512, 1, (1,)),
                                                       nn.Dropout(0.5),
                                                       nn.Sigmoid()) for _ in range(2)])

        self.activation = nn.LeakyReLU(0.2)
        self.action_threshold = args['opt'].action_threshold
        self.background_threshold = args['opt'].background_threshold
        self.similarity_threshold = args['opt'].similarity_threshold
        self.top_k_rat = args['opt'].top_k_rat
        self.weight = 1 / args['opt'].weight

    def forward(self, vfeat, ffeat, **args):
        ori_vatn = self.attentions[0](vfeat)
        ori_fatn = self.attentions[1](ffeat)  # B, 1, T

        action_mask, background_mask, ambiguous_mask, temp_mask, no_action_mask, no_background_mask \
            = self.action_background_mask(ori_vatn, ori_fatn)

        adjacency_action, adjacency_background, adjacency_ambiguous, adjacency_ambiguous_action, adjacency_ambiguous_background, adjacency_ambiguous_ambiguous = self.adjacency_matrix(
            vfeat, ffeat, action_mask, background_mask, ambiguous_mask,
            temp_mask)

        action_vfeat_avg = torch.matmul(vfeat, adjacency_action)
        background_vfeat_avg = torch.matmul(vfeat, adjacency_background)
        ambiguous_vfeat_avg = torch.matmul(vfeat, adjacency_ambiguous)
        vfeat_avg = action_vfeat_avg + background_vfeat_avg + ambiguous_vfeat_avg
        new_vfeat = (vfeat + vfeat_avg) / 2

        action_ffeat_avg = torch.matmul(ffeat, adjacency_action)
        background_ffeat_avg = torch.matmul(ffeat, adjacency_background)
        ambiguous_ffeat_avg = torch.matmul(ffeat, adjacency_ambiguous)
        ffeat_avg = action_ffeat_avg + background_ffeat_avg + ambiguous_ffeat_avg
        new_ffeat = (ffeat + ffeat_avg) / 2

        v_atn = self.attentions[0](new_vfeat)
        f_atn = self.attentions[1](new_ffeat)

        return v_atn, new_vfeat, f_atn, new_ffeat, {}

    def action_background_mask(self, f_atn, v_atn):
        T = f_atn.shape[2]

        action_row_mask = ((f_atn >= self.action_threshold) & (v_atn >= self.action_threshold)).to(torch.float)
        background_row_mask = ((f_atn < self.background_threshold) & (v_atn < self.background_threshold)).to(
            torch.float)

        action_background_row_mask = action_row_mask + background_row_mask
        ambiguous_row_mask = 1 - action_background_row_mask

        ambiguous_mask = torch.matmul(action_background_row_mask.transpose(-1, -2), ambiguous_row_mask)

        action_mask = torch.matmul(action_row_mask.transpose(-1, -2), action_row_mask)
        background_mask = torch.matmul(background_row_mask.transpose(-1, -2), background_row_mask)

        return action_mask, background_mask, ambiguous_mask, \
               action_background_row_mask.repeat(1, T, 1), action_row_mask == 0, background_row_mask == 0

    def adjacency_matrix(self, vfeat, ffeat, action_mask, background_mask, ambiguous_mask, temp_mask):
        """
        features"B,D,T
        """
        B = ffeat.shape[0]
        T = ffeat.shape[2]
        # graph
        f_feat = F.normalize(ffeat, dim=1)
        v_feat = F.normalize(vfeat, dim=1)
        v_similarity = torch.matmul(v_feat.transpose(1, 2), v_feat)
        f_similarity = torch.matmul(f_feat.transpose(1, 2), f_feat)

        fusion_similarity = (v_similarity + f_similarity) / 2

        # mask and normalize
        mask_value = 0
        fusion_similarity[fusion_similarity < self.similarity_threshold] = mask_value

        k = T // self.top_k_rat

        top_k_indices = torch.topk(fusion_similarity, T - k, dim=1, largest=False, sorted=False)[1]
        fusion_similarity = fusion_similarity.scatter(1, top_k_indices, mask_value)

        adjacency_action = fusion_similarity.masked_fill(action_mask == 0, mask_value)
        adjacency_background = fusion_similarity.masked_fill(background_mask == 0, mask_value)
        ambiguous_mask = (ambiguous_mask + torch.eye(T).masked_fill(temp_mask == 1, 0)) == 0
        adjacency_ambiguous = fusion_similarity.masked_fill(ambiguous_mask, mask_value)

        adjacency_action = F.normalize(adjacency_action, p=1, dim=1)
        adjacency_background = F.normalize(adjacency_background, p=1, dim=1)
        adjacency_ambiguous = F.normalize(adjacency_ambiguous, p=1, dim=1)

        ambiguous_action_mask = (action_mask.sum(dim=-1, keepdim=True) == 0).repeat(1, 1, T)
        adjacency_ambiguous_action = adjacency_ambiguous.masked_fill(ambiguous_action_mask, 0)

        ambiguous_background_mask = (background_mask.sum(dim=-1, keepdim=True) == 0).repeat(1, 1, T)
        adjacency_ambiguous_background = adjacency_ambiguous.masked_fill(ambiguous_background_mask, 0)

        adjacency_ambiguous_ambiguous = adjacency_ambiguous.masked_fill(torch.eye(T).unsqueeze(0).repeat(B, 1, 1) == 0,
                                                                        0)

        return adjacency_action, adjacency_background, adjacency_ambiguous, adjacency_ambiguous_action, adjacency_ambiguous_background, adjacency_ambiguous_ambiguous


# DDG_Net ambiguity model (Exp 5 in this paper)
class DDG_Net_ambiguous(nn.Module):
    def __init__(self, n_feature, args):
        super().__init__()

        self.action_graph = nn.ModuleList(
            [nn.ModuleList([nn.Conv1d(n_feature, n_feature, (1,), padding=0) for _ in range(2)]) for _ in range(2)])

        self.background_graph = nn.ModuleList(
            [nn.ModuleList([nn.Conv1d(n_feature, n_feature, (1,), padding=0) for _ in range(2)]) for _ in range(2)])

        self.attentions = nn.ModuleList([nn.Sequential(nn.Conv1d(n_feature, 512, (3,), padding=1),
                                                       nn.LeakyReLU(0.2),
                                                       nn.Dropout(0.5),
                                                       nn.Conv1d(512, 512, (3,), padding=1),
                                                       nn.LeakyReLU(0.2),
                                                       nn.Conv1d(512, 1, (1,)),
                                                       nn.Dropout(0.5),
                                                       nn.Sigmoid()) for _ in range(2)])
        self.activation = nn.LeakyReLU(0.2)
        self.action_threshold = args['opt'].action_threshold
        self.background_threshold = args['opt'].background_threshold
        self.similarity_threshold = args['opt'].similarity_threshold
        self.temperature = args['opt'].temperature
        self.top_k_rat = args['opt'].top_k_rat
        self.weight = 1 / args['opt'].weight

    def forward(self, vfeat, ffeat, **args):
        ori_vatn = self.attentions[0](vfeat)
        ori_fatn = self.attentions[1](ffeat)  # B, 1, T

        action_mask, background_mask, ambiguous_mask, temp_mask, no_action_mask, no_background_mask \
            = self.action_background_mask(ori_vatn, ori_fatn)

        adjacency_action, adjacency_background, adjacency_ambiguous, adjacency_ambiguous_action, adjacency_ambiguous_background, adjacency_ambiguous_ambiguous, \
            = self.adjacency_matrix(vfeat, ffeat, action_mask, background_mask, ambiguous_mask,
                                    temp_mask)

        vfeat_avg = torch.matmul(vfeat, adjacency_action) + torch.matmul(vfeat, adjacency_background) + torch.matmul(
            vfeat, adjacency_ambiguous)

        action_vfeat_gcn = vfeat.clone()
        background_vfeat_gcn = vfeat.clone()
        for layer_a, layer_b in zip(self.action_graph[0], self.background_graph[0]):
            action_vfeat_gcn = self.activation(torch.matmul(layer_a(action_vfeat_gcn), adjacency_action))
            background_vfeat_gcn = self.activation(torch.matmul(layer_b(background_vfeat_gcn), adjacency_background))
        ambiguous_vfeat_gcn = torch.matmul(action_vfeat_gcn, adjacency_ambiguous_action) + \
                              torch.matmul(background_vfeat_gcn, adjacency_ambiguous_background) + \
                              torch.matmul(vfeat, adjacency_ambiguous_ambiguous)
        vfeat_gcn = action_vfeat_gcn + background_vfeat_gcn + ambiguous_vfeat_gcn

        new_vfeat = self.weight * vfeat + (1 - self.weight) * (vfeat_avg + vfeat_gcn) / 2

        ffeat_avg = torch.matmul(ffeat, adjacency_action) + torch.matmul(ffeat, adjacency_background) + torch.matmul(
            ffeat, adjacency_ambiguous)

        action_ffeat_gcn = ffeat.clone()
        background_ffeat_gcn = ffeat.clone()
        for layer_a, layer_b in zip(self.action_graph[1], self.background_graph[1]):
            action_ffeat_gcn = self.activation(torch.matmul(layer_a(action_ffeat_gcn), adjacency_action))
            background_ffeat_gcn = self.activation(torch.matmul(layer_b(background_ffeat_gcn), adjacency_background))
        ambiguous_ffeat_gcn = torch.matmul(action_ffeat_gcn, adjacency_ambiguous_action) + \
                              torch.matmul(background_ffeat_gcn, adjacency_ambiguous_background) + \
                              torch.matmul(ffeat, adjacency_ambiguous_ambiguous)
        ffeat_gcn = action_ffeat_gcn + background_ffeat_gcn + ambiguous_ffeat_gcn

        new_ffeat = self.weight * ffeat + (1 - self.weight) * (ffeat_avg + ffeat_gcn) / 2

        v_atn = self.attentions[0](new_vfeat)
        f_atn = self.attentions[1](new_ffeat)

        if args['is_training']:
            loss = self.cp_loss(ori_vatn, ori_fatn, no_action_mask,
                                no_background_mask, ffeat_avg, ffeat_gcn, vfeat_avg, vfeat_gcn)

            return v_atn, new_vfeat, f_atn, new_ffeat, loss
        else:
            return v_atn, new_vfeat, f_atn, new_ffeat, {}

    def action_background_mask(self, f_atn, v_atn):

        T = f_atn.shape[2]

        action_row_mask = ((f_atn >= self.action_threshold) & (v_atn >= self.action_threshold)).to(torch.float)
        background_row_mask = ((f_atn < self.background_threshold) & (v_atn < self.background_threshold)).to(
            torch.float)

        action_background_row_mask = action_row_mask + background_row_mask
        ambiguous_row_mask = 1 - action_background_row_mask
        all_mask = action_background_row_mask + ambiguous_row_mask

        ambiguous_mask = torch.matmul(all_mask.transpose(-1, -2), ambiguous_row_mask)
        action_mask = torch.matmul((action_row_mask + ambiguous_row_mask).transpose(-1, -2), action_row_mask)
        background_mask = torch.matmul((background_row_mask + ambiguous_row_mask).transpose(-1, -2),
                                       background_row_mask)

        return action_mask, background_mask, ambiguous_mask, \
               action_background_row_mask.repeat(1, T, 1), action_row_mask == 0, background_row_mask == 0

    def adjacency_matrix(self, vfeat, ffeat, action_mask, background_mask, ambiguous_mask, temp_mask):
        """
        features"B,D,T
        """
        B = ffeat.shape[0]
        T = ffeat.shape[2]
        # graph
        f_feat = F.normalize(ffeat, dim=1)
        v_feat = F.normalize(vfeat, dim=1)
        v_similarity = torch.matmul(v_feat.transpose(1, 2), v_feat)
        f_similarity = torch.matmul(f_feat.transpose(1, 2), f_feat)

        fusion_similarity = (v_similarity + f_similarity) / 2

        # mask and normalize
        mask_value = 0

        fusion_similarity[fusion_similarity < self.similarity_threshold] = mask_value

        k = T // self.top_k_rat

        top_k_indices = torch.topk(fusion_similarity, T - k, dim=1, largest=False, sorted=False)[1]
        fusion_similarity = fusion_similarity.scatter(1, top_k_indices, mask_value)

        adjacency_action = fusion_similarity.masked_fill(action_mask == 0, mask_value)
        adjacency_background = fusion_similarity.masked_fill(background_mask == 0, mask_value)
        ambiguous_mask = (ambiguous_mask + torch.eye(T).masked_fill(temp_mask == 1, 0)) == 0
        adjacency_ambiguous = fusion_similarity.masked_fill(ambiguous_mask, mask_value)

        adjacency_action = F.normalize(adjacency_action, p=1, dim=1)
        adjacency_background = F.normalize(adjacency_background, p=1, dim=1)
        adjacency_ambiguous = F.normalize(adjacency_ambiguous, p=1, dim=1)

        ambiguous_action_mask = (action_mask.sum(dim=-1, keepdim=True) == 0).repeat(1, 1, T)
        adjacency_ambiguous_action = adjacency_ambiguous.masked_fill(ambiguous_action_mask, 0)

        ambiguous_background_mask = (background_mask.sum(dim=-1, keepdim=True) == 0).repeat(1, 1, T)
        adjacency_ambiguous_background = adjacency_ambiguous.masked_fill(ambiguous_background_mask, 0)

        adjacency_ambiguous_ambiguous = adjacency_ambiguous.masked_fill(torch.eye(T).unsqueeze(0).repeat(B, 1, 1) == 0,
                                                                        0)

        return adjacency_action, adjacency_background, adjacency_ambiguous, adjacency_ambiguous_action, adjacency_ambiguous_background, adjacency_ambiguous_ambiguous

    def cp_loss(self, ori_v_atn, ori_f_atn, no_action_mask,
                no_background_mask, ffeat_avg, ffeat_gcn, vfeat_avg, vfeat_gcn):

        action_mask = no_action_mask == False
        background_mask = no_background_mask == False

        ori_v_atn = ori_v_atn.detach()
        ori_f_atn = ori_f_atn.detach()
        ori_atn = (ori_f_atn + ori_v_atn) / 2

        # 4
        action_count = action_mask.sum(dim=-1).squeeze()
        background_count = background_mask.sum(dim=-1).squeeze()
        action_count = max(action_count.count_nonzero().item(), 1)
        background_count = max(background_count.count_nonzero().item(), 1)
        feat_loss = 0.5 * (
                (torch.sum(torch.exp(-(1 / ori_atn - 1) / self.temperature).masked_fill(no_action_mask,
                                                                                        0).detach() * torch.norm(
                    vfeat_avg - vfeat_gcn, dim=1, keepdim=True), dim=-1) /
                 torch.exp(-(1 / ori_atn - 1) / self.temperature).masked_fill(no_action_mask, 0).detach().sum(
                     dim=-1).clamp(min=1e-3)).sum() / action_count + \
                (torch.sum(torch.exp(-(1 / (1 - ori_atn) - 1) / self.temperature).masked_fill(no_background_mask,
                                                                                              0).detach() * torch.norm(
                    vfeat_avg - vfeat_gcn, dim=1, keepdim=True), dim=-1) /
                 torch.exp(-(1 / (1 - ori_atn) - 1) / self.temperature).masked_fill(no_background_mask, 0).detach().sum(
                     dim=-1).clamp(min=1e-3)).sum() / background_count) + \
                    0.5 * (
                            (torch.sum(torch.exp(-(1 / ori_atn - 1) / self.temperature).masked_fill(no_action_mask,
                                                                                                    0).detach() * torch.norm(
                                ffeat_avg - ffeat_gcn, dim=1, keepdim=True), dim=-1) /
                             torch.exp(-(1 / ori_atn - 1) / self.temperature).masked_fill(no_action_mask,
                                                                                          0).detach().sum(dim=-1).clamp(
                                 min=1e-3)).sum() / action_count + \
                            (torch.sum(
                                torch.exp(-(1 / (1 - ori_atn) - 1) / self.temperature).masked_fill(no_background_mask,
                                                                                                   0).detach() * torch.norm(
                                    ffeat_avg - ffeat_gcn, dim=1, keepdim=True), dim=-1) /
                             torch.exp(-(1 / (1 - ori_atn) - 1) / self.temperature).masked_fill(no_background_mask,
                                                                                                0).detach().sum(
                                 dim=-1).clamp(min=1e-3)).sum() / background_count)

        return {'feat_loss': feat_loss}


# DDG_Net separate adjacency matrixes for RGB and optical flow (Exp 4 in this paper)
class DDG_Net_separate(nn.Module):
    def __init__(self, n_feature, args):
        super().__init__()

        self.action_graph = nn.ModuleList(
            [nn.ModuleList([nn.Conv1d(n_feature, n_feature, (1,), padding=0) for _ in range(2)]) for _ in range(2)])

        self.background_graph = nn.ModuleList(
            [nn.ModuleList([nn.Conv1d(n_feature, n_feature, (1,), padding=0) for _ in range(2)]) for _ in range(2)])

        self.attentions = nn.ModuleList([nn.Sequential(nn.Conv1d(n_feature, 512, (3,), padding=1),
                                                       nn.LeakyReLU(0.2),
                                                       nn.Dropout(0.5),
                                                       nn.Conv1d(512, 512, (3,), padding=1),
                                                       nn.LeakyReLU(0.2),
                                                       nn.Conv1d(512, 1, (1,)),
                                                       nn.Dropout(0.5),
                                                       nn.Sigmoid()) for _ in range(2)])
        self.activation = nn.LeakyReLU(0.2)
        self.action_threshold = args['opt'].action_threshold
        self.background_threshold = args['opt'].background_threshold
        self.similarity_threshold = args['opt'].similarity_threshold
        self.temperature = args['opt'].temperature
        self.top_k_rat = args['opt'].top_k_rat
        self.weight = 1 / args['opt'].weight

    def forward(self, vfeat, ffeat, **args):
        ori_vatn = self.attentions[0](vfeat)
        ori_fatn = self.attentions[1](ffeat)  # B, 1, T

        action_mask, background_mask, ambiguous_mask, temp_mask, no_action_mask, no_background_mask \
            = self.action_background_mask(ori_vatn, ori_fatn)

        adjacency_action_v, adjacency_background_v, adjacency_ambiguous_v, adjacency_ambiguous_action_v, adjacency_ambiguous_background_v, adjacency_ambiguous_ambiguous_v, \
        adjacency_action_f, adjacency_background_f, adjacency_ambiguous_f, adjacency_ambiguous_action_f, adjacency_ambiguous_background_f, adjacency_ambiguous_ambiguous_f, \
            = self.adjacency_matrix(vfeat, ffeat, action_mask, background_mask, ambiguous_mask,
                                    temp_mask)

        vfeat_avg = torch.matmul(vfeat, adjacency_action_v) + torch.matmul(vfeat,
                                                                           adjacency_background_v) + torch.matmul(vfeat,
                                                                                                                  adjacency_ambiguous_v)

        action_vfeat_gcn = vfeat.clone()
        background_vfeat_gcn = vfeat.clone()
        for layer_a, layer_b in zip(self.action_graph[0], self.background_graph[0]):
            action_vfeat_gcn = self.activation(torch.matmul(layer_a(action_vfeat_gcn), adjacency_action_v))
            background_vfeat_gcn = self.activation(torch.matmul(layer_b(background_vfeat_gcn), adjacency_background_v))
        ambiguous_vfeat_gcn = torch.matmul(action_vfeat_gcn, adjacency_ambiguous_action_v) + \
                              torch.matmul(background_vfeat_gcn, adjacency_ambiguous_background_v) + \
                              torch.matmul(vfeat, adjacency_ambiguous_ambiguous_v)
        vfeat_gcn = action_vfeat_gcn + background_vfeat_gcn + ambiguous_vfeat_gcn

        new_vfeat = self.weight * vfeat + (1 - self.weight) * (vfeat_avg + vfeat_gcn) / 2

        ffeat_avg = torch.matmul(ffeat, adjacency_action_f) + torch.matmul(ffeat,
                                                                           adjacency_background_f) + torch.matmul(ffeat,
                                                                                                                  adjacency_ambiguous_f)

        action_ffeat_gcn = ffeat.clone()
        background_ffeat_gcn = ffeat.clone()
        for layer_a, layer_b in zip(self.action_graph[1], self.background_graph[1]):
            action_ffeat_gcn = self.activation(torch.matmul(layer_a(action_ffeat_gcn), adjacency_action_f))
            background_ffeat_gcn = self.activation(torch.matmul(layer_b(background_ffeat_gcn), adjacency_background_f))
        ambiguous_ffeat_gcn = torch.matmul(action_ffeat_gcn, adjacency_ambiguous_action_f) + \
                              torch.matmul(background_ffeat_gcn, adjacency_ambiguous_background_f) + \
                              torch.matmul(ffeat, adjacency_ambiguous_ambiguous_f)
        ffeat_gcn = action_ffeat_gcn + background_ffeat_gcn + ambiguous_ffeat_gcn

        new_ffeat = self.weight * ffeat + (1 - self.weight) * (ffeat_avg + ffeat_gcn) / 2

        v_atn = self.attentions[0](new_vfeat)
        f_atn = self.attentions[1](new_ffeat)

        if args['is_training']:
            loss = self.cp_loss(ori_vatn, ori_fatn, no_action_mask,
                                no_background_mask, ffeat_avg, ffeat_gcn, vfeat_avg, vfeat_gcn)

            return v_atn, new_vfeat, f_atn, new_ffeat, loss
        else:
            return v_atn, new_vfeat, f_atn, new_ffeat, {}

    def action_background_mask(self, f_atn, v_atn):

        T = f_atn.shape[2]

        action_row_mask = ((f_atn >= self.action_threshold) & (v_atn >= self.action_threshold)).to(torch.float)
        background_row_mask = ((f_atn < self.background_threshold) & (v_atn < self.background_threshold)).to(
            torch.float)

        action_background_row_mask = action_row_mask + background_row_mask
        ambiguous_row_mask = 1 - action_background_row_mask

        ambiguous_mask = torch.matmul(action_background_row_mask.transpose(-1, -2), ambiguous_row_mask)
        action_mask = torch.matmul(action_row_mask.transpose(-1, -2), action_row_mask)
        background_mask = torch.matmul(background_row_mask.transpose(-1, -2), background_row_mask)

        return action_mask, background_mask, ambiguous_mask, \
               action_background_row_mask.repeat(1, T, 1), action_row_mask == 0, background_row_mask == 0

    def adjacency_matrix(self, vfeat, ffeat, action_mask, background_mask, ambiguous_mask, temp_mask):
        """
        features"B,D,T
        """
        B = ffeat.shape[0]
        T = ffeat.shape[2]
        # graph
        f_feat = F.normalize(ffeat, dim=1)
        v_feat = F.normalize(vfeat, dim=1)
        v_similarity = torch.matmul(v_feat.transpose(1, 2), v_feat)
        f_similarity = torch.matmul(f_feat.transpose(1, 2), f_feat)

        # mask and normalize
        mask_value = 0
        v_similarity[v_similarity < self.similarity_threshold] = mask_value
        f_similarity[f_similarity < self.similarity_threshold] = mask_value

        k = T // self.top_k_rat

        top_k_indices = torch.topk(v_similarity, T - k, dim=1, largest=False, sorted=False)[1]
        v_similarity = v_similarity.scatter(1, top_k_indices, mask_value)
        top_k_indices = torch.topk(f_similarity, T - k, dim=1, largest=False, sorted=False)[1]
        f_similarity = f_similarity.scatter(1, top_k_indices, mask_value)

        adjacency_action_v = v_similarity.masked_fill(action_mask == 0, mask_value)
        adjacency_background_v = v_similarity.masked_fill(background_mask == 0, mask_value)
        ambiguous_mask = (ambiguous_mask + torch.eye(T).masked_fill(temp_mask == 1, 0)) == 0
        adjacency_ambiguous_v = v_similarity.masked_fill(ambiguous_mask, mask_value)

        adjacency_action_f = f_similarity.masked_fill(action_mask == 0, mask_value)
        adjacency_background_f = f_similarity.masked_fill(background_mask == 0, mask_value)
        ambiguous_mask = (ambiguous_mask + torch.eye(T).masked_fill(temp_mask == 1, 0)) == 0
        adjacency_ambiguous_f = f_similarity.masked_fill(ambiguous_mask, mask_value)

        adjacency_action_v = F.normalize(adjacency_action_v, p=1, dim=1)
        adjacency_background_v = F.normalize(adjacency_background_v, p=1, dim=1)
        adjacency_ambiguous_v = F.normalize(adjacency_ambiguous_v, p=1, dim=1)

        adjacency_action_f = F.normalize(adjacency_action_f, p=1, dim=1)
        adjacency_background_f = F.normalize(adjacency_background_f, p=1, dim=1)
        adjacency_ambiguous_f = F.normalize(adjacency_ambiguous_f, p=1, dim=1)

        ambiguous_action_mask = (action_mask.sum(dim=-1, keepdim=True) == 0).repeat(1, 1, T)
        adjacency_ambiguous_action_v = adjacency_ambiguous_v.masked_fill(ambiguous_action_mask, 0)

        ambiguous_background_mask = (background_mask.sum(dim=-1, keepdim=True) == 0).repeat(1, 1, T)
        adjacency_ambiguous_background_v = adjacency_ambiguous_v.masked_fill(ambiguous_background_mask, 0)

        adjacency_ambiguous_ambiguous_v = adjacency_ambiguous_v.masked_fill(
            torch.eye(T).unsqueeze(0).repeat(B, 1, 1) == 0, 0)

        ambiguous_action_mask = (action_mask.sum(dim=-1, keepdim=True) == 0).repeat(1, 1, T)
        adjacency_ambiguous_action_f = adjacency_ambiguous_f.masked_fill(ambiguous_action_mask, 0)

        ambiguous_background_mask = (background_mask.sum(dim=-1, keepdim=True) == 0).repeat(1, 1, T)
        adjacency_ambiguous_background_f = adjacency_ambiguous_f.masked_fill(ambiguous_background_mask, 0)

        adjacency_ambiguous_ambiguous_f = adjacency_ambiguous_f.masked_fill(
            torch.eye(T).unsqueeze(0).repeat(B, 1, 1) == 0, 0)

        return adjacency_action_v, adjacency_background_v, adjacency_ambiguous_v, adjacency_ambiguous_action_v, adjacency_ambiguous_background_v, adjacency_ambiguous_ambiguous_v, \
               adjacency_action_f, adjacency_background_f, adjacency_ambiguous_f, adjacency_ambiguous_action_f, adjacency_ambiguous_background_f, adjacency_ambiguous_ambiguous_f

    def cp_loss(self, ori_v_atn, ori_f_atn, no_action_mask,
                no_background_mask, ffeat_avg, ffeat_gcn, vfeat_avg, vfeat_gcn):

        action_mask = no_action_mask == False
        background_mask = no_background_mask == False

        ori_v_atn = ori_v_atn.detach()
        ori_f_atn = ori_f_atn.detach()
        ori_atn = (ori_f_atn + ori_v_atn) / 2

        # 4
        action_count = action_mask.sum(dim=-1).squeeze()
        background_count = background_mask.sum(dim=-1).squeeze()
        action_count = max(action_count.count_nonzero().item(), 1)
        background_count = max(background_count.count_nonzero().item(), 1)
        feat_loss = 0.5 * (
                (torch.sum(torch.exp(-(1 / ori_atn - 1) / self.temperature).masked_fill(no_action_mask,
                                                                                        0).detach() * torch.norm(
                    vfeat_avg - vfeat_gcn, dim=1, keepdim=True), dim=-1) /
                 torch.exp(-(1 / ori_atn - 1) / self.temperature).masked_fill(no_action_mask, 0).detach().sum(
                     dim=-1).clamp(min=1e-3)).sum() / action_count + \
                (torch.sum(torch.exp(-(1 / (1 - ori_atn) - 1) / self.temperature).masked_fill(no_background_mask,
                                                                                              0).detach() * torch.norm(
                    vfeat_avg - vfeat_gcn, dim=1, keepdim=True), dim=-1) /
                 torch.exp(-(1 / (1 - ori_atn) - 1) / self.temperature).masked_fill(no_background_mask, 0).detach().sum(
                     dim=-1).clamp(min=1e-3)).sum() / background_count) + \
                    0.5 * (
                            (torch.sum(torch.exp(-(1 / ori_atn - 1) / self.temperature).masked_fill(no_action_mask,
                                                                                                    0).detach() * torch.norm(
                                ffeat_avg - ffeat_gcn, dim=1, keepdim=True), dim=-1) /
                             torch.exp(-(1 / ori_atn - 1) / self.temperature).masked_fill(no_action_mask,
                                                                                          0).detach().sum(dim=-1).clamp(
                                 min=1e-3)).sum() / action_count + \
                            (torch.sum(
                                torch.exp(-(1 / (1 - ori_atn) - 1) / self.temperature).masked_fill(no_background_mask,
                                                                                                   0).detach() * torch.norm(
                                    ffeat_avg - ffeat_gcn, dim=1, keepdim=True), dim=-1) /
                             torch.exp(-(1 / (1 - ori_atn) - 1) / self.temperature).masked_fill(no_background_mask,
                                                                                                0).detach().sum(
                                 dim=-1).clamp(min=1e-3)).sum() / background_count)

        return {'feat_loss': feat_loss}


# Ours
class DDG_Net(nn.Module):
    def __init__(self, n_feature, args):
        super().__init__()

        self.action_graph = nn.ModuleList(
            [nn.ModuleList([nn.Conv1d(n_feature, n_feature, (1,), padding=0) for _ in range(2)]) for _ in range(2)])

        self.background_graph = nn.ModuleList(
            [nn.ModuleList([nn.Conv1d(n_feature, n_feature, (1,), padding=0) for _ in range(2)]) for _ in range(2)])

        self.attentions = nn.ModuleList([nn.Sequential(nn.Conv1d(n_feature, 512, (3,), padding=1),
                                                       nn.LeakyReLU(0.2),
                                                       nn.Dropout(0.5),
                                                       nn.Conv1d(512, 512, (3,), padding=1),
                                                       nn.LeakyReLU(0.2),
                                                       nn.Conv1d(512, 1, (1,)),
                                                       nn.Dropout(0.5),
                                                       nn.Sigmoid()) for _ in range(2)])
        self.activation = nn.LeakyReLU(0.2)
        self.action_threshold = args['opt'].action_threshold
        self.background_threshold = args['opt'].background_threshold
        self.similarity_threshold = args['opt'].similarity_threshold
        self.temperature = args['opt'].temperature
        self.top_k_rat = args['opt'].top_k_rat
        self.weight = 1 / args['opt'].weight

    def forward(self, vfeat, ffeat, **args):
        ori_vatn = self.attentions[0](vfeat)
        ori_fatn = self.attentions[1](ffeat)  # B, 1, T

        action_mask, background_mask, ambiguous_mask, temp_mask, no_action_mask, no_background_mask \
            = self.action_background_mask(ori_vatn, ori_fatn)

        adjacency_action, adjacency_background, adjacency_ambiguous, adjacency_ambiguous_action, adjacency_ambiguous_background, adjacency_ambiguous_ambiguous, \
            = self.adjacency_matrix(vfeat, ffeat, action_mask, background_mask, ambiguous_mask,
                                    temp_mask)

        vfeat_avg = torch.matmul(vfeat, adjacency_action) + torch.matmul(vfeat, adjacency_background) + torch.matmul(
            vfeat, adjacency_ambiguous)

        action_vfeat_gcn = vfeat.clone()
        background_vfeat_gcn = vfeat.clone()
        for layer_a, layer_b in zip(self.action_graph[0], self.background_graph[0]):
            action_vfeat_gcn = self.activation(torch.matmul(layer_a(action_vfeat_gcn), adjacency_action))
            background_vfeat_gcn = self.activation(torch.matmul(layer_b(background_vfeat_gcn), adjacency_background))
        ambiguous_vfeat_gcn = torch.matmul(action_vfeat_gcn, adjacency_ambiguous_action) + \
                              torch.matmul(background_vfeat_gcn, adjacency_ambiguous_background) + \
                              torch.matmul(vfeat, adjacency_ambiguous_ambiguous)
        vfeat_gcn = action_vfeat_gcn + background_vfeat_gcn + ambiguous_vfeat_gcn

        new_vfeat = self.weight * vfeat + (1 - self.weight) * (vfeat_avg + vfeat_gcn) / 2

        ffeat_avg = torch.matmul(ffeat, adjacency_action) + torch.matmul(ffeat, adjacency_background) + torch.matmul(
            ffeat, adjacency_ambiguous)

        action_ffeat_gcn = ffeat.clone()
        background_ffeat_gcn = ffeat.clone()
        for layer_a, layer_b in zip(self.action_graph[1], self.background_graph[1]):
            action_ffeat_gcn = self.activation(torch.matmul(layer_a(action_ffeat_gcn), adjacency_action))
            background_ffeat_gcn = self.activation(torch.matmul(layer_b(background_ffeat_gcn), adjacency_background))
        ambiguous_ffeat_gcn = torch.matmul(action_ffeat_gcn, adjacency_ambiguous_action) + \
                              torch.matmul(background_ffeat_gcn, adjacency_ambiguous_background) + \
                              torch.matmul(ffeat, adjacency_ambiguous_ambiguous)
        ffeat_gcn = action_ffeat_gcn + background_ffeat_gcn + ambiguous_ffeat_gcn

        new_ffeat = self.weight * ffeat + (1 - self.weight) * (ffeat_avg + ffeat_gcn) / 2

        v_atn = self.attentions[0](new_vfeat)
        f_atn = self.attentions[1](new_ffeat)

        if args['is_training']:
            loss = self.cp_loss(ori_vatn, ori_fatn, no_action_mask,
                                no_background_mask, ffeat_avg, ffeat_gcn, vfeat_avg, vfeat_gcn)

            return v_atn, new_vfeat, f_atn, new_ffeat, loss
        else:
            return v_atn, new_vfeat, f_atn, new_ffeat, {}

    def action_background_mask(self, f_atn, v_atn):

        T = f_atn.shape[2]

        action_row_mask = ((f_atn >= self.action_threshold) & (v_atn >= self.action_threshold)).to(torch.float)
        background_row_mask = ((f_atn < self.background_threshold) & (v_atn < self.background_threshold)).to(
            torch.float)

        action_background_row_mask = action_row_mask + background_row_mask
        ambiguous_row_mask = 1 - action_background_row_mask

        ambiguous_mask = torch.matmul(action_background_row_mask.transpose(-1, -2), ambiguous_row_mask)
        action_mask = torch.matmul(action_row_mask.transpose(-1, -2), action_row_mask)
        background_mask = torch.matmul(background_row_mask.transpose(-1, -2), background_row_mask)

        return action_mask, background_mask, ambiguous_mask, \
               action_background_row_mask.repeat(1, T, 1), action_row_mask == 0, background_row_mask == 0

    def adjacency_matrix(self, vfeat, ffeat, action_mask, background_mask, ambiguous_mask, temp_mask):
        """
        features"B,D,T
        """
        B = ffeat.shape[0]
        T = ffeat.shape[2]
        # graph
        f_feat = F.normalize(ffeat, dim=1)
        v_feat = F.normalize(vfeat, dim=1)
        v_similarity = torch.matmul(v_feat.transpose(1, 2), v_feat)
        f_similarity = torch.matmul(f_feat.transpose(1, 2), f_feat)

        fusion_similarity = (v_similarity + f_similarity) / 2

        # mask and normalize
        mask_value = 0
        fusion_similarity[fusion_similarity < self.similarity_threshold] = mask_value

        k = T // self.top_k_rat

        top_k_indices = torch.topk(fusion_similarity, T - k, dim=1, largest=False, sorted=False)[1]
        fusion_similarity = fusion_similarity.scatter(1, top_k_indices, mask_value)

        adjacency_action = fusion_similarity.masked_fill(action_mask == 0, mask_value)
        adjacency_background = fusion_similarity.masked_fill(background_mask == 0, mask_value)
        ambiguous_mask = (ambiguous_mask + torch.eye(T).masked_fill(temp_mask == 1, 0)) == 0
        adjacency_ambiguous = fusion_similarity.masked_fill(ambiguous_mask, mask_value)

        adjacency_action = F.normalize(adjacency_action, p=1, dim=1)
        adjacency_background = F.normalize(adjacency_background, p=1, dim=1)
        adjacency_ambiguous = F.normalize(adjacency_ambiguous, p=1, dim=1)

        ambiguous_action_mask = (action_mask.sum(dim=-1, keepdim=True) == 0).repeat(1, 1, T)
        adjacency_ambiguous_action = adjacency_ambiguous.masked_fill(ambiguous_action_mask, 0)

        ambiguous_background_mask = (background_mask.sum(dim=-1, keepdim=True) == 0).repeat(1, 1, T)
        adjacency_ambiguous_background = adjacency_ambiguous.masked_fill(ambiguous_background_mask, 0)

        adjacency_ambiguous_ambiguous = adjacency_ambiguous.masked_fill(torch.eye(T).unsqueeze(0).repeat(B, 1, 1) == 0,
                                                                        0)

        return adjacency_action, adjacency_background, adjacency_ambiguous, adjacency_ambiguous_action, adjacency_ambiguous_background, adjacency_ambiguous_ambiguous

    def cp_loss(self, ori_v_atn, ori_f_atn, no_action_mask,
                no_background_mask, ffeat_avg, ffeat_gcn, vfeat_avg, vfeat_gcn):

        action_mask = no_action_mask == False
        background_mask = no_background_mask == False

        ori_v_atn = ori_v_atn.detach()
        ori_f_atn = ori_f_atn.detach()
        ori_atn = (ori_f_atn + ori_v_atn) / 2

        # 4
        action_count = action_mask.sum(dim=-1).squeeze()
        background_count = background_mask.sum(dim=-1).squeeze()
        action_count = max(action_count.count_nonzero().item(), 1)
        background_count = max(background_count.count_nonzero().item(), 1)
        feat_loss = 0.5 * (
                (torch.sum(torch.exp(-(1 / ori_atn - 1) / self.temperature).masked_fill(no_action_mask,
                                                                                        0).detach() * torch.norm(
                    vfeat_avg - vfeat_gcn, dim=1, keepdim=True), dim=-1) /
                 torch.exp(-(1 / ori_atn - 1) / self.temperature).masked_fill(no_action_mask, 0).detach().sum(
                     dim=-1).clamp(min=1e-3)).sum() / action_count + \
                (torch.sum(torch.exp(-(1 / (1 - ori_atn) - 1) / self.temperature).masked_fill(no_background_mask,
                                                                                              0).detach() * torch.norm(
                    vfeat_avg - vfeat_gcn, dim=1, keepdim=True), dim=-1) /
                 torch.exp(-(1 / (1 - ori_atn) - 1) / self.temperature).masked_fill(no_background_mask, 0).detach().sum(
                     dim=-1).clamp(min=1e-3)).sum() / background_count) + \
                    0.5 * (
                            (torch.sum(torch.exp(-(1 / ori_atn - 1) / self.temperature).masked_fill(no_action_mask,
                                                                                                    0).detach() * torch.norm(
                                ffeat_avg - ffeat_gcn, dim=1, keepdim=True), dim=-1) /
                             torch.exp(-(1 / ori_atn - 1) / self.temperature).masked_fill(no_action_mask,
                                                                                          0).detach().sum(dim=-1).clamp(
                                 min=1e-3)).sum() / action_count + \
                            (torch.sum(
                                torch.exp(-(1 / (1 - ori_atn) - 1) / self.temperature).masked_fill(no_background_mask,
                                                                                                   0).detach() * torch.norm(
                                    ffeat_avg - ffeat_gcn, dim=1, keepdim=True), dim=-1) /
                             torch.exp(-(1 / (1 - ori_atn) - 1) / self.temperature).masked_fill(no_background_mask,
                                                                                                0).detach().sum(
                                 dim=-1).clamp(min=1e-3)).sum() / background_count)

        return {'feat_loss': feat_loss}


class DELU(torch.nn.Module):
    def __init__(self, n_feature, n_class, **args):
        super().__init__()
        embed_dim = 2048
        dropout_ratio = args['opt'].dropout_ratio

        self.Attn = getattr(model, args['opt'].AWM)(1024, args)

        self.fusion = nn.Sequential(
            nn.Conv1d(n_feature, n_feature, (1,), padding=0),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_ratio)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout_ratio),
            nn.Conv1d(embed_dim, embed_dim, (3,), padding=1),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.7),
            nn.Conv1d(embed_dim, n_class + 1, (1,))
        )

        self.apply(weights_init)

    def forward(self, inputs, **args):
        feat = inputs.transpose(-1, -2)
        v_atn, vfeat, f_atn, ffeat, loss = self.Attn(feat[:, :1024, :], feat[:, 1024:, :], **args)
        x_atn = (f_atn + v_atn) / 2
        tfeat = torch.cat((vfeat, ffeat), 1)
        nfeat = self.fusion(tfeat)
        x_cls = self.classifier(nfeat)

        outputs = {
            'feat': nfeat.transpose(-1, -2),
            'cas': x_cls.transpose(-1, -2),
            'attn': x_atn.transpose(-1, -2),
            'v_atn': v_atn.transpose(-1, -2),
            'f_atn': f_atn.transpose(-1, -2),
            'extra_loss': loss,
        }

        return outputs

    def _multiply(self, x, atn, dim=-1, include_min=False):
        if include_min:
            _min = x.min(dim=dim, keepdim=True)[0]
        else:
            _min = 0
        return atn * (x - _min) + _min

    def complementary_learning_loss(self, cas, labels):

        labels_with_back = torch.cat(
            (labels, torch.ones_like(labels[:, [0]])), dim=-1).unsqueeze(1)
        cas = F.softmax(cas, dim=-1)
        complementary_loss = torch.sum(-(1 - labels_with_back) * torch.log((1 - cas).clamp_(1e-6)), dim=-1)
        return complementary_loss.mean()

    def criterion(self, outputs, labels, **args):

        feat, element_logits, element_atn = outputs['feat'], outputs['cas'], outputs['attn']
        v_atn = outputs['v_atn']
        f_atn = outputs['f_atn']

        try:
            fc_loss = outputs['extra_loss']['feat_loss']
        except:
            fc_loss = 0

        mutual_loss = 0.5 * \
                      F.mse_loss(v_atn, f_atn.detach()) + 0.5 * \
                      F.mse_loss(f_atn, v_atn.detach())

        element_logits_supp = self._multiply(
            element_logits, element_atn, include_min=True)

        cl_loss = self.complementary_learning_loss(element_logits, labels)

        edl_loss = self.edl_loss(element_logits_supp,
                                 element_atn,
                                 labels,
                                 rat=args['opt'].rat_atn,
                                 n_class=args['opt'].num_class,
                                 epoch=args['itr'],
                                 total_epoch=args['opt'].max_iter,
                                 )

        uct_guide_loss = self.uct_guide_loss(element_logits,
                                             element_logits_supp,
                                             element_atn,
                                             v_atn,
                                             f_atn,
                                             n_class=args['opt'].num_class,
                                             epoch=args['itr'],
                                             total_epoch=args['opt'].max_iter,
                                             amplitude=args['opt'].amplitude,
                                             )

        loss_mil_orig, _ = self.topkloss(element_logits,
                                         labels,
                                         is_back=True,
                                         rat=args['opt'].k)

        # SAL
        loss_mil_supp, _ = self.topkloss(element_logits_supp,
                                         labels,
                                         is_back=False,
                                         rat=args['opt'].k)

        loss_3_supp_Contrastive = self.Contrastive(
            feat, element_logits_supp, labels, is_back=False)

        loss_norm = element_atn.mean()
        # guide loss
        loss_guide = (1 - element_atn -
                      element_logits.softmax(-1)[..., [-1]]).abs().mean()

        v_loss_norm = v_atn.mean()
        # guide loss
        v_loss_guide = (1 - v_atn -
                        element_logits.softmax(-1)[..., [-1]]).abs().mean()

        f_loss_norm = f_atn.mean()
        # guide loss
        f_loss_guide = (1 - f_atn -
                        element_logits.softmax(-1)[..., [-1]]).abs().mean()

        total_loss = (
                args['opt'].alpha_edl * edl_loss +
                args['opt'].alpha_uct_guide * uct_guide_loss +
                loss_mil_orig.mean() + loss_mil_supp.mean() +
                args['opt'].alpha3 * loss_3_supp_Contrastive +
                args['opt'].alpha4 * mutual_loss +
                args['opt'].alpha1 * (loss_norm + v_loss_norm + f_loss_norm) / 3 +
                args['opt'].alpha2 * (loss_guide + v_loss_guide + f_loss_guide) / 3
                + args['opt'].alpha5 * fc_loss
                + args['opt'].alpha6 * cl_loss
        )

        loss_dict = {
            'edl_loss': args['opt'].alpha_edl * edl_loss,
            'uct_guide_loss': args['opt'].alpha_uct_guide * uct_guide_loss,
            'loss_mil_orig': loss_mil_orig.mean(),
            'loss_mil_supp': loss_mil_supp.mean(),
            'loss_supp_contrastive': args['opt'].alpha3 * loss_3_supp_Contrastive,
            'mutual_loss': args['opt'].alpha4 * mutual_loss,
            'norm_loss': args['opt'].alpha1 * (loss_norm + v_loss_norm + f_loss_norm) / 3,
            'guide_loss': args['opt'].alpha2 * (loss_guide + v_loss_guide + f_loss_guide) / 3,
            'feat_loss': args['opt'].alpha5 * fc_loss,
            'complementary_loss': args['opt'].alpha6 * cl_loss,
            'total_loss': total_loss,
        }

        return total_loss, loss_dict

    def uct_guide_loss(self,
                       element_logits,
                       element_logits_supp,
                       element_atn,
                       v_atn,
                       f_atn,
                       n_class,
                       epoch,
                       total_epoch,
                       amplitude):

        evidence = exp_evidence(element_logits_supp)
        alpha = evidence + 1
        S = torch.sum(alpha, dim=-1)
        snippet_uct = n_class / S

        total_snippet_num = element_logits.shape[1]
        curve = self.course_function(
            epoch, total_epoch, total_snippet_num, amplitude)

        loss_guide = (1 - element_atn - element_logits.softmax(-1)
        [..., [-1]]).abs().squeeze()

        v_loss_guide = (1 - v_atn - element_logits.softmax(-1)
        [..., [-1]]).abs().squeeze()

        f_loss_guide = (1 - f_atn - element_logits.softmax(-1)
        [..., [-1]]).abs().squeeze()

        total_loss_guide = (loss_guide + v_loss_guide + f_loss_guide) / 3

        _, uct_indices = torch.sort(snippet_uct, dim=1)
        sorted_curve = torch.gather(curve.repeat(10, 1), 1, uct_indices)

        uct_guide_loss = torch.mul(sorted_curve, total_loss_guide).mean()

        return uct_guide_loss

    def edl_loss(self,
                 element_logits_supp,
                 element_atn,
                 labels,
                 rat,
                 n_class,
                 epoch=0,
                 total_epoch=5000,
                 ):

        k = max(1, int(element_logits_supp.shape[-2] // rat))

        atn_values, atn_idx = torch.topk(
            element_atn,
            k=k,
            dim=1
        )
        atn_idx_expand = atn_idx.expand([-1, -1, n_class + 1])
        topk_element_logits = torch.gather(
            element_logits_supp, 1, atn_idx_expand)[:, :, :-1]
        video_logits = topk_element_logits.mean(dim=1)

        edl_loss = EvidenceLoss(
            num_classes=n_class,
            evidence='exp',
            loss_type='log',
            with_kldiv=False,
            with_avuloss=False,
            disentangle=False,
            annealing_method='exp')

        edl_results = edl_loss(
            output=video_logits,
            target=labels,
            epoch=epoch,
            total_epoch=total_epoch
        )

        edl_loss = edl_results['loss_cls'].mean()

        return edl_loss

    def course_function(self, epoch, total_epoch, total_snippet_num, amplitude):

        idx = torch.arange(total_snippet_num)
        theta = 2 * (idx + 0.5) / total_snippet_num - 1
        delta = - 2 * epoch / total_epoch + 1
        curve = amplitude * torch.tanh(theta * delta) + 1

        return curve

    def topkloss(self,
                 element_logits,
                 labels,
                 is_back=True,
                 rat=8):

        if is_back:
            labels_with_back = torch.cat(
                (labels, torch.ones_like(labels[:, [0]])), dim=-1)
        else:
            labels_with_back = torch.cat(
                (labels, torch.zeros_like(labels[:, [0]])), dim=-1)

        topk_val, topk_ind = torch.topk(
            element_logits,
            k=max(1, int(element_logits.shape[-2] // rat)),
            dim=-2)

        instance_logits = torch.mean(topk_val, dim=-2)

        labels_with_back = labels_with_back / (
                torch.sum(labels_with_back, dim=1, keepdim=True) + 1e-4)

        milloss = - (labels_with_back *
                     F.log_softmax(instance_logits, dim=-1)).sum(dim=-1)

        return milloss, topk_ind

    def Contrastive(self, x, element_logits, labels, is_back=False):
        if is_back:
            labels = torch.cat(
                (labels, torch.ones_like(labels[:, [0]])), dim=-1)
        else:
            labels = torch.cat(
                (labels, torch.zeros_like(labels[:, [0]])), dim=-1)
        sim_loss = 0.
        n_tmp = 0.
        _, n, c = element_logits.shape
        for i in range(0, 3 * 2, 2):
            atn1 = F.softmax(element_logits[i], dim=0)
            atn2 = F.softmax(element_logits[i + 1], dim=0)

            n1 = torch.FloatTensor([np.maximum(n - 1, 1)]).cuda()
            n2 = torch.FloatTensor([np.maximum(n - 1, 1)]).cuda()
            # (n_feature, n_class)
            Hf1 = torch.mm(torch.transpose(x[i], 1, 0), atn1)
            Hf2 = torch.mm(torch.transpose(x[i + 1], 1, 0), atn2)
            Lf1 = torch.mm(torch.transpose(x[i], 1, 0), (1 - atn1) / n1)
            Lf2 = torch.mm(torch.transpose(x[i + 1], 1, 0), (1 - atn2) / n2)

            d1 = 1 - torch.sum(Hf1 * Hf2, dim=0) / (
                    torch.norm(Hf1, 2, dim=0) * torch.norm(Hf2, 2, dim=0))  # 1-similarity
            d2 = 1 - torch.sum(Hf1 * Lf2, dim=0) / \
                 (torch.norm(Hf1, 2, dim=0) * torch.norm(Lf2, 2, dim=0))
            d3 = 1 - torch.sum(Hf2 * Lf1, dim=0) / \
                 (torch.norm(Hf2, 2, dim=0) * torch.norm(Lf1, 2, dim=0))
            sim_loss = sim_loss + 0.5 * torch.sum(
                torch.max(d1 - d2 + 0.5, torch.FloatTensor([0.]).cuda()) * labels[i, :] * labels[i + 1, :])
            sim_loss = sim_loss + 0.5 * torch.sum(
                torch.max(d1 - d3 + 0.5, torch.FloatTensor([0.]).cuda()) * labels[i, :] * labels[i + 1, :])
            n_tmp = n_tmp + torch.sum(labels[i, :] * labels[i + 1, :])
        sim_loss = sim_loss / n_tmp
        return sim_loss

    def decompose(self, outputs, **args):
        feat, element_logits, atn_supp, atn_drop, element_atn = outputs

        return element_logits, element_atn


class DELU_ACT(torch.nn.Module):
    def __init__(self, n_feature, n_class, **args):
        super().__init__()
        embed_dim = 2048
        mid_dim = 1024
        dropout_ratio = args['opt'].dropout_ratio
        reduce_ratio = args['opt'].reduce_ratio

        self.Attn = getattr(model, args['opt'].AWM)(1024, args)

        self.feat_encoder = nn.Sequential(
            nn.Conv1d(n_feature, embed_dim, 3, padding=1), nn.LeakyReLU(0.2), nn.Dropout(dropout_ratio))
        self.fusion = nn.Sequential(
            nn.Conv1d(n_feature, n_feature, 1, padding=0), nn.LeakyReLU(0.2), nn.Dropout(dropout_ratio))
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_ratio),
            nn.Conv1d(embed_dim, embed_dim, 3, padding=1), nn.LeakyReLU(0.2),
            nn.Dropout(0.7), nn.Conv1d(embed_dim, n_class + 1, 1))
        # self.cadl = CADL()
        # self.attention = Non_Local_Block(embed_dim,mid_dim,dropout_ratio)

        self.channel_avg = nn.AdaptiveAvgPool1d(1)
        self.batch_avg = nn.AdaptiveAvgPool1d(1)
        self.ce_criterion = nn.BCELoss()
        _kernel = ((args['opt'].max_seqlen // args['opt'].t) // 2 * 2 + 1)
        self.pool = nn.AvgPool1d(_kernel, 1, padding=_kernel // 2, count_include_pad=True) \
            if _kernel is not None else nn.Identity()
        self.apply(weights_init)

    def forward(self, inputs, **args):
        feat = inputs.transpose(-1, -2)
        b, c, n = feat.size()
        # feat = self.feat_encoder(x)
        v_atn, vfeat, f_atn, ffeat, loss = self.Attn(feat[:, :1024, :], feat[:, 1024:, :], **args)
        x_atn = (f_atn + v_atn) / 2
        nfeat = torch.cat((vfeat, ffeat), 1)
        nfeat = self.fusion(nfeat)
        x_cls = self.classifier(nfeat)
        x_cls = self.pool(x_cls)
        x_atn = self.pool(x_atn)
        f_atn = self.pool(f_atn)
        v_atn = self.pool(v_atn)
        # fg_mask, bg_mask,dropped_fg_mask = self.cadl(x_cls, x_atn, include_min=True)

        return {'feat': nfeat.transpose(-1, -2), 'cas': x_cls.transpose(-1, -2), 'attn': x_atn.transpose(-1, -2),
                'v_atn': v_atn.transpose(-1, -2), 'f_atn': f_atn.transpose(-1, -2), 'extra_loss': loss}
        # ,fg_mask.transpose(-1, -2), bg_mask.transpose(-1, -2),dropped_fg_mask.transpose(-1, -2)
        # return att_sigmoid,att_logit, feat_emb, bag_logit, instance_logit

    def _multiply(self, x, atn, dim=-1, include_min=False):
        if include_min:
            _min = x.min(dim=dim, keepdim=True)[0]
        else:
            _min = 0
        return atn * (x - _min) + _min

    def complementary_learning_loss(self, cas, labels):
        labels_with_back = torch.cat(
            (labels, torch.ones_like(labels[:, [0]])), dim=-1).unsqueeze(1)
        cas = F.softmax(cas, dim=-1)
        complementary_loss = torch.sum(-(1 - labels_with_back) * torch.log((1 - cas).clamp_(1e-6)), dim=-1)
        return complementary_loss.mean()

    def criterion(self, outputs, labels, **args):
        feat, element_logits, element_atn = outputs['feat'], outputs['cas'], outputs['attn']
        v_atn = outputs['v_atn']
        f_atn = outputs['f_atn']

        try:
            fc_loss = outputs['extra_loss']['feat_loss']
        except:
            fc_loss = 0

        mutual_loss = 0.5 * F.mse_loss(v_atn, f_atn.detach()) + 0.5 * F.mse_loss(f_atn, v_atn.detach())
        # learning weight dynamic, lambda1 (1-lambda1)
        b, n, c = element_logits.shape
        cl_loss = self.complementary_learning_loss(element_logits, labels)
        element_logits_supp = self._multiply(element_logits, element_atn, include_min=True)
        loss_mil_orig, _ = self.topkloss(element_logits,
                                         labels,
                                         is_back=True,
                                         rat=args['opt'].k,
                                         reduce=None)
        # SAL
        loss_mil_supp, _ = self.topkloss(element_logits_supp,
                                         labels,
                                         is_back=False,
                                         rat=args['opt'].k,
                                         reduce=None)

        edl_loss = self.edl_loss(element_logits_supp,
                                 element_atn,
                                 labels,
                                 rat=args['opt'].rat_atn,
                                 n_class=args['opt'].num_class,
                                 epoch=args['itr'],
                                 total_epoch=args['opt'].max_iter,
                                 )

        uct_guide_loss = self.uct_guide_loss(element_logits,
                                             element_logits_supp,
                                             element_atn,
                                             v_atn,
                                             f_atn,
                                             n_class=args['opt'].num_class,
                                             epoch=args['itr'],
                                             total_epoch=args['opt'].max_iter,
                                             amplitude=args['opt'].amplitude,
                                             )

        loss_3_supp_Contrastive = self.Contrastive(feat, element_logits_supp, labels, is_back=False)

        loss_norm = element_atn.mean()
        # guide loss
        loss_guide = (1 - element_atn -
                      element_logits.softmax(-1)[..., [-1]]).abs().mean()

        v_loss_norm = v_atn.mean()
        # guide loss
        v_loss_guide = (1 - v_atn -
                        element_logits.softmax(-1)[..., [-1]]).abs().mean()

        f_loss_norm = f_atn.mean()
        # guide loss
        f_loss_guide = (1 - f_atn -
                        element_logits.softmax(-1)[..., [-1]]).abs().mean()

        # total loss
        total_loss = (
                args['opt'].alpha_edl * edl_loss +
                args['opt'].alpha_uct_guide * uct_guide_loss +
                loss_mil_orig.mean() + loss_mil_supp.mean() +
                args['opt'].alpha3 * loss_3_supp_Contrastive +
                args['opt'].alpha4 * mutual_loss +
                args['opt'].alpha1 * (loss_norm + v_loss_norm + f_loss_norm) / 3 +
                args['opt'].alpha2 * (loss_guide + v_loss_guide + f_loss_guide) / 3
                + args['opt'].alpha5 * fc_loss
                + args['opt'].alpha6 * cl_loss
        )

        loss_dict = {
            'edl_loss': args['opt'].alpha_edl * edl_loss,
            'uct_guide_loss': args['opt'].alpha_uct_guide * uct_guide_loss,
            'loss_mil_orig': loss_mil_orig.mean(),
            'loss_mil_supp': loss_mil_supp.mean(),
            'loss_supp_contrastive': args['opt'].alpha3 * loss_3_supp_Contrastive,
            'mutual_loss': args['opt'].alpha4 * mutual_loss,
            'norm_loss': args['opt'].alpha1 * (loss_norm + v_loss_norm + f_loss_norm) / 3,
            'guide_loss': args['opt'].alpha2 * (loss_guide + v_loss_guide + f_loss_guide) / 3,
            'total_loss': total_loss,
            'feat_loss': args['opt'].alpha5 * fc_loss,
            'complementary_loss': args['opt'].alpha6 * cl_loss,
        }

        return total_loss, loss_dict

    def uct_guide_loss(self,
                       element_logits,
                       element_logits_supp,
                       element_atn,
                       v_atn,
                       f_atn,
                       n_class,
                       epoch,
                       total_epoch,
                       amplitude):

        evidence = exp_evidence(element_logits_supp)
        alpha = evidence + 1
        S = torch.sum(alpha, dim=-1)
        snippet_uct = n_class / S

        total_snippet_num = element_logits.shape[1]
        curve = self.course_function(epoch, total_epoch, total_snippet_num, amplitude)

        loss_guide = (1 - element_atn - element_logits.softmax(-1)[..., [-1]]).abs().squeeze()

        v_loss_guide = (1 - v_atn - element_logits.softmax(-1)[..., [-1]]).abs().squeeze()

        f_loss_guide = (1 - f_atn - element_logits.softmax(-1)[..., [-1]]).abs().squeeze()

        total_loss_guide = (loss_guide + v_loss_guide + f_loss_guide) / 3

        _, uct_indices = torch.sort(snippet_uct, dim=1)
        sorted_curve = torch.gather(curve.repeat(10, 1), 1, uct_indices)

        uct_guide_loss = torch.mul(sorted_curve, total_loss_guide).mean()

        return uct_guide_loss

    def edl_loss(self,
                 element_logits_supp,
                 element_atn,
                 labels,
                 rat,
                 n_class,
                 epoch=0,
                 total_epoch=5000,
                 ):

        k = max(1, int(element_logits_supp.shape[-2] // rat))

        atn_values, atn_idx = torch.topk(
            element_atn,
            k=k,
            dim=1
        )

        atn_idx_expand = atn_idx.expand([-1, -1, n_class + 1])
        topk_element_logits = torch.gather(element_logits_supp, 1, atn_idx_expand)[:, :, :-1]

        video_logits = topk_element_logits.mean(dim=1)

        edl_loss = EvidenceLoss(
            num_classes=n_class,
            evidence='relu',
            loss_type='mse',
            with_kldiv=False,
            with_avuloss=False,
            disentangle=False,
            annealing_method='exp')

        edl_results = edl_loss(
            output=video_logits,
            target=labels,
            epoch=epoch,
            total_epoch=total_epoch
        )

        edl_loss = edl_results['loss_cls'].mean()

        return edl_loss

    def course_function(self, epoch, total_epoch, total_snippet_num, amplitude):

        idx = torch.arange(total_snippet_num)

        # From -1 to 1
        theta = 2 * (idx + 0.5) / total_snippet_num - 1

        # From 1 to -1
        delta = - 2 * epoch / total_epoch + 1

        curve = amplitude * torch.tanh(theta * delta) + 1

        return curve

    def topkloss(self,
                 element_logits,
                 labels,
                 is_back=True,
                 lab_rand=None,
                 rat=8,
                 reduce=None):

        if is_back:
            labels_with_back = torch.cat(
                (labels, torch.ones_like(labels[:, [0]])), dim=-1)
        else:
            labels_with_back = torch.cat(
                (labels, torch.zeros_like(labels[:, [0]])), dim=-1)
        if lab_rand is not None:
            labels_with_back = torch.cat((labels, lab_rand), dim=-1)

        topk_val, topk_ind = torch.topk(
            element_logits,
            k=max(1, int(element_logits.shape[-2] // rat)),
            dim=-2)
        instance_logits = torch.mean(
            topk_val,
            dim=-2,
        )
        labels_with_back = labels_with_back / (
                torch.sum(labels_with_back, dim=1, keepdim=True) + 1e-4)
        milloss = (-(labels_with_back *
                     F.log_softmax(instance_logits, dim=-1)).sum(dim=-1))
        if reduce is not None:
            milloss = milloss.mean()
        return milloss, topk_ind

    def Contrastive(self, x, element_logits, labels, is_back=False):
        if is_back:
            labels = torch.cat(
                (labels, torch.ones_like(labels[:, [0]])), dim=-1)
        else:
            labels = torch.cat(
                (labels, torch.zeros_like(labels[:, [0]])), dim=-1)
        sim_loss = 0.
        n_tmp = 0.
        _, n, c = element_logits.shape
        for i in range(0, 3 * 2, 2):
            atn1 = F.softmax(element_logits[i], dim=0)
            atn2 = F.softmax(element_logits[i + 1], dim=0)

            n1 = torch.FloatTensor([np.maximum(n - 1, 1)]).cuda()
            n2 = torch.FloatTensor([np.maximum(n - 1, 1)]).cuda()
            Hf1 = torch.mm(torch.transpose(x[i], 1, 0), atn1)  # (n_feature, n_class)
            Hf2 = torch.mm(torch.transpose(x[i + 1], 1, 0), atn2)
            Lf1 = torch.mm(torch.transpose(x[i], 1, 0), (1 - atn1) / n1)
            Lf2 = torch.mm(torch.transpose(x[i + 1], 1, 0), (1 - atn2) / n2)

            d1 = 1 - torch.sum(Hf1 * Hf2, dim=0) / (
                    torch.norm(Hf1, 2, dim=0) * torch.norm(Hf2, 2, dim=0))  # 1-similarity
            d2 = 1 - torch.sum(Hf1 * Lf2, dim=0) / (torch.norm(Hf1, 2, dim=0) * torch.norm(Lf2, 2, dim=0))
            d3 = 1 - torch.sum(Hf2 * Lf1, dim=0) / (torch.norm(Hf2, 2, dim=0) * torch.norm(Lf1, 2, dim=0))
            sim_loss = sim_loss + 0.5 * torch.sum(
                torch.max(d1 - d2 + 0.5, torch.FloatTensor([0.]).cuda()) * labels[i, :] * labels[i + 1, :])
            sim_loss = sim_loss + 0.5 * torch.sum(
                torch.max(d1 - d3 + 0.5, torch.FloatTensor([0.]).cuda()) * labels[i, :] * labels[i + 1, :])
            n_tmp = n_tmp + torch.sum(labels[i, :] * labels[i + 1, :])
        sim_loss = sim_loss / n_tmp
        return sim_loss

    def decompose(self, outputs, **args):
        feat, element_logits, atn_supp, atn_drop, element_atn = outputs

        return element_logits, element_atn

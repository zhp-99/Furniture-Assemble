import torch.nn as nn
import torch
import torch.nn.functional as F
from models.pointnet.pointnet2_utils import PointNetSetAbstraction,PointNetFeaturePropagation

class PointNet2SemSegSSG(nn.Module):
    def __init__(self):
        super(PointNet2SemSegSSG, self).__init__()
        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointNetSetAbstraction(
                npoint=1024,
                radius=0.1,
                nsample=32,
                in_channel=3 + 3,
                mlp=[32, 32, 64],
                group_all=False
            )
        )
        self.SA_modules.append(
            PointNetSetAbstraction(
                npoint=256,
                radius=0.2,
                nsample=32,
                in_channel=64 + 3,
                mlp=[64, 64, 128],
                group_all=False
            )
        )
        self.SA_modules.append(
            PointNetSetAbstraction(
                npoint=64,
                radius=0.4,
                nsample=32,
                in_channel=128 + 3,
                mlp=[128, 128, 256],
                group_all=False
            )
        )

        self.SA_modules.append(
            PointNetSetAbstraction(
                npoint=16,
                radius=0.8,
                nsample=32,
                in_channel=256 + 3,
                mlp=[256, 256, 512],
                group_all=False
            )
        )

        self.FP_modules = nn.ModuleList()
        self.FP_modules.append(PointNetFeaturePropagation(in_channel=128 + 3, mlp=[128, 128, 128]))
        self.FP_modules.append(PointNetFeaturePropagation(in_channel=256 + 64, mlp=[256, 128]))
        self.FP_modules.append(PointNetFeaturePropagation(in_channel=256 + 128, mlp=[256, 256]))
        self.FP_modules.append(PointNetFeaturePropagation(in_channel=512 + 256, mlp=[256, 256]))
        
        self.fc_layer = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
        )

    def forward(self, points):

        B,C,N = points.shape
        features = points
        xyz = points[:,:3,:]

        l_xyz, l_features = [xyz], [features]
        for i, layer in enumerate(self.SA_modules):
            li_xyz, li_features = layer(l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)
        
        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](
                l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i]
            )

        return self.fc_layer(l_features[0])

class Critic(nn.Module):

    def __init__(self, input_dim, output_dim=1, hidden_dim=128):
        super(Critic, self).__init__()

        self.hidden_dim = hidden_dim
        self.mlp1 = nn.Linear(input_dim, self.hidden_dim)
        self.mlp2 = nn.Linear(self.hidden_dim, output_dim)

    # pixel_feats B x F, query_fats: B x 6
    # output: B
    def forward(self, inputs):
        input_net = torch.cat(inputs, dim=-1)
        hidden_net = F.leaky_relu(self.mlp1(input_net))
        net = self.mlp2(hidden_net)
        return net

class Network(nn.Module):
    def __init__(self, feat_dim, cp_feat_dim, dir_feat_dim, hidden_feat_dim=128):
        super(Network, self).__init__()

        self.pointnet2 = PointNet2SemSegSSG()

        self.critic = Critic(input_dim=feat_dim + cp_feat_dim + dir_feat_dim, hidden_dim=hidden_feat_dim)

        self.mlp_dir = nn.Linear(3, dir_feat_dim)
        self.mlp_cp = nn.Linear(3, cp_feat_dim)     # contact point

        self.BCELoss = nn.BCEWithLogitsLoss(reduction='none')
        self.sigmoid = nn.Sigmoid()
        self.BCELoss_withoutSigmoid = nn.BCELoss(reduction='none')
        self.L1Loss = nn.L1Loss(reduction='none')

    def forward(self, pcs, cp1, dir1):
        pcs[:, 0, :] = cp1

        pcs = pcs.transpose(1, 2)

        whole_feats = self.pointnet2(pcs)

        # feature for contact point
        net1 = whole_feats[:, :, 0]

        cp1_feats = self.mlp_cp(cp1)
        dir1_feats = self.mlp_dir(dir1)

        pred_result_logits = self.critic([net1, cp1_feats, dir1_feats])
        pred_scores = 0.1*torch.sigmoid(pred_result_logits)
    
        return pred_scores.squeeze(1)


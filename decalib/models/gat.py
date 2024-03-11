import face_alignment
import torch
import torch.nn as nn
import torch.nn.functional as F


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, noutput, dropout, alpha, nheads, batchsize):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout
        self.batchsize = batchsize

        """GAT related modules"""
        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True, batchsize=self.batchsize) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.out_att = GraphAttentionLayer(nhid * nheads, nhid, dropout=dropout, alpha=alpha, concat=False, batchsize=self.batchsize)

        """Patch extraction related modules"""
        self.fan = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)

        """Decoder related modules"""
        self.transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(nfeat, nhead=4),
            num_layers=2
        )
        self.fc = nn.Sequential(
            nn.Linear(nfeat,64),
            nn.ReLU(),
            nn.Linear(64, noutput)
        )

    def extract_patch(self, image, patch_size=32):
        half_patch_size = patch_size // 2
        img = image.permute(0,2,3,1)[0] * 255.
        lmks = self.fan.get_landmarks(img)
        img = img / 255.
        if len(lmks) > 1:
            lmks = lmks[0]
        patches = []
        for x, y in lmks:
            x = int(x); y = int(y)
            patch = img[0,y - half_patch_size:y + half_patch_size,
                      x - half_patch_size:x + half_patch_size]
            patches.append(patch)
        return patches
        
        
    def forward(self, x, adj=None):
        x = self.extract_patch(x)
        if adj == None:
            adj = torch.ones((x.shape[0], x.shape[1], x.shape[1])).to(x.device)
        # x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=2)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.out_att(x, adj)
        x = self.transformer(x)
        x = self.fc(x)
        return x


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout, alpha, batchsize, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.batchsize = batchsize
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(self.batchsize, self.in_features, self.out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(self.batchsize, 2*self.out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.mm(h, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)
        e = self._prepare_attentional_mechanism_input(Wh)

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        Wh1 = torch.matmul(Wh, self.a[:,:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[:,self.out_features:, :])
        # broadcast add
        Wh2T = Wh2.permute(0, 2, 1)
        e = Wh1 + Wh2T
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
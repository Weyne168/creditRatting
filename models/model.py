import torch
import torch.nn as nn
import math


class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class CrossAttn(nn.Module):
    def __init__(self, in_dim, seq_len=5, ch_factor=2):
        super(CrossAttn, self).__init__()
        self.chanel_in = in_dim
        self.confuse_hi = nn.Conv2d(in_channels=seq_len, out_channels=1, kernel_size=1, bias=False)
        self.confuse_x0 = nn.Conv2d(in_channels=seq_len, out_channels=1, kernel_size=1, bias=False)
        # self.confuse_vh = nn.Conv2d(in_channels=seq_len, out_channels=1, kernel_size=1, bias=False)

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // ch_factor, kernel_size=1, bias=False)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // ch_factor, kernel_size=1, bias=False)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1, bias=False)
        # self.value_xf = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // ch_factor, kernel_size=1, bias=False)
        # self.confuse = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1, bias=False)

        self.softmax = nn.Softmax(dim=-1)
        self.activate = nn.PReLU()
        self.seq_len = seq_len
        self.dropout = nn.Dropout(0.1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, hi, x0):
        '''
        :param hi: batch, seq_len, rf, dim
        :param x0: batch, seq_len, nf, dim
        :return:
        '''
        # print(hi.size(), x0.size())
        B, rf = hi.size(0), hi.size(2)
        nf = x0.size(2)
        hf = self.confuse_hi(hi)
        xf = self.confuse_x0(x0)

        xf = xf.permute(0, 3, 1, 2)  # b,dim,1,nf
        hf = hf.permute(0, 3, 1, 2)  # b,dim,1,rf
        # xf = xf.contiguous()

        query = self.query_conv(xf).squeeze()  # B,d1,nf
        query = query.permute(0, 2, 1)  # B,nf,d1
        key = self.key_conv(hf).squeeze()  # B,d1,rf

        energy = torch.bmm(query, key)  # B,rf,nf
        attention = self.softmax(energy)  # B,rf,nf
        attention = self.dropout(attention)
        attention_copy = attention.unsqueeze(1).repeat(1, self.seq_len, 1, 1)
        # attention_copy = attention_copy.view(B, nf, rf)
        attention_copy = attention_copy.unsqueeze(-1)

        value = hi.unsqueeze(2) * x0.unsqueeze(3)  # B,s,rf,nf,d
        # print(value.size(), attention_copy.size())
        # out = self.activate(value * attention_copy) + value
        out = self.gamma * value * attention_copy + value
        out = out.view(B, self.seq_len, rf * nf, -1)
        return out, attention.squeeze()


class FeatureEmbedding(nn.Module):
    def __init__(self, device, n_num_feats, cls_feats=[2], dim=32, nm=True):
        super(FeatureEmbedding, self).__init__()
        self.cls_encoders = []
        for ncls in cls_feats:
            self.cls_encoders.append(nn.Embedding(ncls, dim).to(device))
        self.num_encoder = nn.Embedding(n_num_feats, dim)
        self.LayerNorm = BertLayerNorm(dim)
        self.dropout = nn.Dropout(0.1)
        self.nm = nm

    def forward(self, xc, xn):
        '''
        :param xc: category feature, shape of batch, seq_len, feats
        :param xn: numberical feature, shape of batch, seq_len, feats
        :return:
        '''
        # categorical features embedding
        # print(xc.size(),xn.size())
        # exit(2)
        b, s, d = xc.size()
        xc_embs = []
        for i in range(d):
            _embed = self.cls_encoders[i](xc[:, :, i].long())
            xc_embs.append(_embed.unsqueeze(2))
        xc_embed = torch.cat(xc_embs, dim=2)
        # numberical features embedding
        xn = xn.unsqueeze(-1)
        xn = xn * self.num_encoder.weight

        xnc = torch.cat((xc_embed, xn), dim=2)
        xnc = self.dropout(xnc)
        if self.nm:
            xnc = self.LayerNorm(xnc)
        # output shape: batch, sequence_len, n_feat, dim
        return xnc


class FeatureCrossing(nn.Module):
    '''
    K: the number of output high-order features
    '''

    def __init__(self, device, n0_feats, nr_feats, seq_len, nr1_feats, K=16):
        super(FeatureCrossing, self).__init__()
        self.pfs = nn.Conv2d(in_channels=n0_feats * nr_feats, out_channels=nr1_feats, kernel_size=1, bias=False).to(
            device)
        self.cross_layer = CrossAttn(K, seq_len).to(device)

    def forward(self, xr, x0):
        '''
        :param x0: the first order feature embeds
        :param xr: the r-order feature embeds
        :return:
        '''
        hi, attention = self.cross_layer(xr, x0)
        # b,s,rf,d->b,rf,s,d
        hi = hi.permute(0, 2, 1, 3)
        hi = hi.contiguous()
        # print(hi.size())
        hi = self.pfs(hi)
        # b,rf,s,d->b,s,rf,d
        hi = hi.permute(0, 2, 1, 3)
        hi = hi.contiguous()
        return hi, attention, self.pfs.weight


class FeatureAttention(nn.Module):
    def __init__(self, seq_len, embed_dim):
        super(FeatureAttention, self).__init__()
        self.scores_ = nn.Sequential(
            nn.Conv2d(in_channels=seq_len, out_channels=seq_len, kernel_size=(1, embed_dim), bias=False)
            # nn.Sigmoid()
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        b, s, nf, d = x.size()
        x = self.scores_(x)  # b,s,nf,1
        x = x.view(b, -1)
        x = self.softmax(x)
        x = x.view(b, s, nf, 1)
        return x


class CNNDLGA(nn.Module):
    def __init__(self, device, seq_len, n_num_feats, cls_feats=[2], embed_dim=64,
                 keep_cross_feats=[128, 64, 32], nclass=19):
        super(CNNDLGA, self).__init__()
        self.seq_len = seq_len
        self.featEmbedding = FeatureEmbedding(device, n_num_feats, cls_feats, embed_dim)
        self.cross_layers = []
        n0_feats = n_num_feats + len(cls_feats)
        for i, rf in enumerate(keep_cross_feats):
            if i == 0:
                cross_layer = FeatureCrossing(device, n0_feats, n0_feats, seq_len, rf, embed_dim)
            else:  # device, n0_feats, nr_feats, seq_len, K=16
                cross_layer = FeatureCrossing(device, n0_feats, keep_cross_feats[i - 1], seq_len, rf, embed_dim)
            self.cross_layers.append(cross_layer)

        total_num_feats = sum(keep_cross_feats) + n0_feats
        # self.global_gru = nn.GRU(input_size=total_num_feats * embed_dim, hidden_size=128, num_layers=2, dropout=0.5, bidirectional=True, batch_first=True)
        self.feature_atten = FeatureAttention(seq_len, embed_dim)
        self.dropout = nn.Dropout(0.1)
        self.fcLayer = nn.Sequential(
            nn.Linear(total_num_feats * embed_dim * seq_len, 256),
            nn.PReLU(),
            nn.Linear(256, nclass)
        )
        self.gamma = nn.Parameter(torch.zeros(1))
        # self.dropout0 = nn.Dropout2d(0.5)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
        self.criterion = nn.CrossEntropyLoss()
        # self.criterion2 = nn.MSELoss()

    def _forward(self, xc, xn):
        b = xc.size(0)
        x0 = self.featEmbedding(xc, xn)
        hs, ws = [], []
        for i, cross_layer in enumerate(self.cross_layers):
            if i == 0:
                hi, attention, fws = cross_layer(x0, x0)
            else:
                hi, attention, fws = cross_layer(hi, x0)
            hs.append(hi)
            ws.append((attention, fws))
        hs.append(x0)
        X = torch.cat(hs, dim=2)
        '''
        gru_in = X.view(b, self.seq_len, -1)
        print('in', gru_in.size())
        gout, hout = self.global_gru(gru_in)
        print('xxxxxxxxxx', gout.size())
        exit(2)
        '''
        feat_scores = self.feature_atten(X)
        X = self.gamma * feat_scores * X + X
        X = self.dropout(X)
        X = X.view(b, -1)
        out = self.fcLayer(X)
        # out0 = F.softmax(out)
        return out, feat_scores, ws

    def forward(self, xc, xn, label=None):
        out, feat_scores, ws = self._forward(xc, xn)
        if self.training:
            # print(out.size(),label.size())
            loss = self.criterion(out, label)
            # print(torch.sum(F.softmax(out)))
            # loss2 = (1.0 - torch.pow(torch.sum(F.softmax(out,dim=1) * label, dim=1), 0.7)) / 0.7
            # loss2 = torch.sum(loss2) / label.size(0)
            return out, loss, None
        else:
            return out, feat_scores, ws


if __name__ == '__main__':
    net = nn.Embedding(3, 7)
    input = torch.ones((2, 5, 3))
    input[1, 3, 0] = 2.5
    input[1, 2, 1] = 1.0
    input[0, 1, 2] = 0.5
    b, s, d = input.size()
    input = input.unsqueeze(-1)
    w = net.weight
    print(w.size(), input.size())
    output = input * w

    print(output.size())

    net = nn.Conv2d(in_channels=5, out_channels=2, kernel_size=1, bias=False)
    print(net.weight.size())

    x0 = torch.ones((4, 5, 3, 8)).unsqueeze(2)
    hi = torch.ones((4, 5, 6, 8)).unsqueeze(3)
    # hi = hi.permute(0, 2, 1, 3)
    hi = hi.contiguous()
    H = hi * x0
    print(H.size())

    hc = torch.randint(2, (2, 3, 2))
    hn = torch.randn((2, 3, 7))
    device = torch.device('cpu')
    net = CNNDLGA(device, 3, 7, cls_feats=[2, 2], embed_dim=64, keep_cross_feats=[16, 8, 4], nclass=19)
    r = net(hc, hn)
    print(r)

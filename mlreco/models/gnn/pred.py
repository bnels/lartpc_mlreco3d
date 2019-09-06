from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, Sigmoid, LeakyReLU, Dropout, BatchNorm1d, Bilinear, Identity

# Edge prediction modules

# final prediction layer
class EdgePredModel(torch.nn.Module):
    def __init__(self, node_in, edge_in, leak, nout=2, bn_momentum=0.1):
        """
        Basic model for making edge predictions
        
        parameters:
            node_in - number of node features coming in
            edge_in - number of edge features coming in
            leak - leakiness of leakyrelus
            nout - number of classes out (default 2)
        """
        super(EdgePredModel, self).__init__()

        self.edge_pred_mlp = Seq(Lin(2*node_in + edge_in, 64),
                                 LeakyReLU(leak),
                                 BatchNorm1d(64, momentum=bn_momentum),
                                 Lin(64, 32),
                                 LeakyReLU(leak),
                                 BatchNorm1d(32, momentum=bn_momentum),
                                 Lin(32, 16),
                                 LeakyReLU(leak),
                                 BatchNorm1d(16, momentum=bn_momentum),
                                 Lin(16,8),
                                 LeakyReLU(leak),
                                 BatchNorm1d(8, momentum=bn_momentum),
                                 Lin(8,nout)
                                )

    def forward(self, src, dest, edge_attr, u, batch):
        return self.edge_pred_mlp(torch.cat([src, dest, edge_attr], dim=1))
    

class FastEdgePredModel(torch.nn.Module):
    def __init__(self, node_in, edge_in, leak, nout=2, bn_momentum=0.1):
        """
        Model for making edge predicitons without many hidden variables
        """
        super(FastEdgePredModel, self).__init__()
        
        self.edge_pred_mlp = Seq(Lin(edge_in, nout*4),
                                 LeakyReLU(leak),
                                 BatchNorm1d(nout*4, momentum=bn_momentum),
                                 Lin(nout*4, nout)
                                )
        def forward(src, dest, edge_attr, u, batch):
            return sefl.edge_pred_mlp(edge_attr)
    
    
class BilinEdgePredModel(torch.nn.Module):
    def __init__(self, node_in, edge_in, leak, nout=2, bn_momentum=0.1):
        """
        Bilinear model for making edge predictions
        
        parameters:
            node_in - number of node features coming in
            edge_in - number of edge features coming in
            leak - leakiness of leakyrelus
            nout - number of classes out (default 2)
        """
        super(BilinEdgePredModel, self).__init__()
        
        self.bse = Bilinear(node_in, edge_in, 16, bias=True)
        self.bte = Bilinear(node_in, edge_in, 16, bias=True)
        self.bst = Bilinear(node_in, node_in, edge_in, bias=False)
        self.bee = Bilinear(edge_in, edge_in, 16, bias=True)
        
        self.mlp = Seq(
            Lin(3*16, 64),
            LeakyReLU(leak),
            Lin(64, 64),
            LeakyReLU(leak),
            Lin(64,32),
            LeakyReLU(leak),
            Lin(32,16),
            LeakyReLU(leak),
            Lin(16, nout)
        )
        
    def forward(self, source, target, edge_attr, u, batch):
        # two bilinear forms
        x = self.bse(source, edge_attr)
        y = self.bte(target, edge_attr)
        
        # trilinear form
        z = self.bst(source, target)
        z = self.bee(z, edge_attr)
        
        out = torch.cat([x, y, z], dim=1)
        out = self.mlp(out)
        return out
    

class CombEdgePredModel(torch.nn.Module):
    def __init__(self, node_in, edge_in, leak, nout=2, nfeat=32, use_bn=True, bn_momentum=0.1):
        """
        Combined model for making edge predictions
        
        parameters:
            node_in - number of node features coming in
            edge_in - number of edge features coming in
            leak - leakiness of leakyrelus
            nout - number of classes out (default 2)
        """
        super(CombEdgePredModel, self).__init__()
        
        
        self.bilin = Bilinear(node_in, edge_in, nfeat, bias=True)
        self.elin = Lin(edge_in, nfeat)
        self.nlin = Lin(node_in, nfeat)
        
        self.mlp = Seq(
            BatchNorm1d(nfeat) if use_bn else Identity(),
            Lin(nfeat, nfeat*2),
            LeakyReLU(leak),
            BatchNorm1d(nfeat*2) if use_bn else Identity(),
            Lin(nfeat*2, nfeat),
            LeakyReLU(leak),
            BatchNorm1d(nfeat) if use_bn else Identity(),
            Lin(nfeat,nfeat//2),
            LeakyReLU(leak),
            BatchNorm1d(nfeat//2) if use_bn else Identity(),
            Lin(nfeat//2, nfeat//4),
            LeakyReLU(leak),
            BatchNorm1d(nfeat//4) if use_bn else Identity(),
            Lin(nfeat//4, nout)
        )
        
    def forward(self, source, target, edge_attr, u, batch):
        # two bilinear forms
        x = self.bilin(source, edge_attr) + self.bilin(target, edge_attr)
        # add edge features
        x = x + self.elin(edge_attr)
        # add node features
        x = x + self.nlin(source) + self.nlin(target)
        
        # pass through MLP
        return self.mlp(x)

    
class NodePredModel(torch.nn.Module):
    def __init__(self, node_in, edge_in, leak, nout=2, bn_momentum=0.1):
        """
        Basic model for making node predictions
        
        parameters:
            node_in - number of node features coming in
            edge_in - number of edge features coming in
            leak - leakiness of leakyrelus
            nout - number of classes out (default 2)
        """
        super(NodePredModel, self).__init__()

        self.pred_mlp = Seq(Lin(node_in, 32),
                             LeakyReLU(leak),
                             BatchNorm1d(32, momentum=bn_momentum),
                             Lin(32, 32),
                             LeakyReLU(leak),
                             BatchNorm1d(32, momentum=bn_momentum),
                             Lin(32, 16),
                             LeakyReLU(leak),
                             BatchNorm1d(16, momentum=bn_momentum),
                             Lin(16,nout)
                            )

    def forward(self, x, edge_index, edge_attr, u, batch):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        return self.pred_mlp(x)
    
    
class FastNodePredModel(torch.nn.Module):
    def __init__(self, node_in, edge_in, leak, nout=2, bn_momentum=0.1):
        """
        Basic model for making node predictions
        
        parameters:
            node_in - number of node features coming in
            edge_in - number of edge features coming in
            leak - leakiness of leakyrelus
            nout - number of classes out (default 2)
        """
        super(FastNodePredModel, self).__init__()

        self.pred_mlp = Seq(Lin(node_in, 4*nout),
                             LeakyReLU(leak),
                             BatchNorm1d(nout*4, momentum=bn_momentum),
                             Lin(4*nout, nout)
                            )

    def forward(self, x, edge_index, edge_attr, u, batch):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        return self.pred_mlp(x)
    
    
    
    
    
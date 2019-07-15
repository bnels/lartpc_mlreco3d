# GNN that attempts to match clusters to groups
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import numpy as np
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, Sigmoid, LeakyReLU, Dropout, BatchNorm1d
from torch_geometric.nn import MetaLayer, NNConv

class NNConvModel(torch.nn.Module):
    """
    Simple GNN with several edge convolutions, followed by MetLayer for edge prediction
    
    for use in config
    model:
        modules:
            attention_gnn:
                nheads: <number of heads for attention>
    """
    def __init__(self, cfg):
        super(NNConvModel, self).__init__()
        
        
        if 'modules' in cfg:
            self.model_config = cfg['modules']['attention_gnn']
        else:
            self.model_config = cfg
            
        
        self.aggr = self.model_config.get('aggr', 'add')
        self.leak = self.model_config.get('leak', 0.1)
        
        # perform batch normalization
        self.bn_node = BatchNorm1d(16)
        self.bn_edge = BatchNorm1d(10)
        
        # go from 16 to 32 node features
        ninput = 16
        noutput = 32
        self.nn1 = Seq(
            Lin(10, ninput),
            LeakyReLU(self.leak),
            Lin(ninput, ninput*noutput),
            LeakyReLU(self.leak)
        )
        self.layer1 = NNConv(ninput, noutput, self.nn1, aggr=self.aggr)
        
        # go from 32 to 64 node features
        ninput = 32
        noutput = 64
        self.nn2 = Seq(
            Lin(10, ninput),
            LeakyReLU(self.leak),
            Lin(ninput, ninput*noutput),
            LeakyReLU(self.leak)
        )
        self.layer2 = NNConv(ninput, noutput, self.nn2, aggr=self.aggr)
        
        # final prediction layer
        self.edge_pred_mlp = Seq(Lin(138, 64), LeakyReLU(self.leak), Lin(64, 16), LeakyReLU(self.leak), Lin(16,1), Sigmoid())
        
        def edge_pred_model(source, target, edge_attr, u, batch):
            out = torch.cat([source, target, edge_attr], dim=1)
            out = self.edge_pred_mlp(out)
            return out
        
        self.edge_predictor = MetaLayer(edge_pred_model, None, None)
        
        
    def forward(self, x, edge_index, e, xbatch):
        """
        inputs data:
            x - vertex features
            edge_index - graph edge list
            e - edge features
            xbatch - node batchid
        """
        
        # batch normalization of node features
        x = self.bn_node(x)
        # batch normalization of edge features
        e = self.bn_edge(e)
        
        # go through layers
        x = self.layer1(x, edge_index, e)

        x = self.layer2(x, edge_index, e)
        
        x, e, u = self.edge_predictor(x, edge_index, e, u=None, batch=xbatch)

        return {
            'edge_pred': e
        }
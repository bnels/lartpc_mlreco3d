# GNN that attempts to match clusters to groups
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import numpy as np
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, Sigmoid, LeakyReLU, Dropout, BatchNorm1d, Bilinear
from torch_geometric.nn import MetaLayer, NNConv

from .pred import EdgePredModel, BilinEdgePredModel, NodePredModel, CombEdgePredModel

class EdgeModel(torch.nn.Module):
    def __init__(self, node_in, edge_in, edge_out, leak, bn_momentum):
        super(EdgeModel, self).__init__()

        nin = 2*node_in + edge_in
        nout = max(2*nin, 32)
        # residual - type network
        self.nn = Seq(
            Lin(nin, nout),
            LeakyReLU(leak, inplace=True),
            BatchNorm1d(nout, momentum=bn_momentum),
            Lin(nout, edge_out),
            LeakyReLU(leak, inplace=True),
        )
        self.lin = Lin(edge_in, edge_out)
        #self.bilin = Bilinear(node_in, edge_in, edge_out, bias=True)
        self.bn = BatchNorm1d(edge_out, momentum=bn_momentum)
        
    def forward(self, source, target, edge_attr, u, batch):
        
        x = torch.cat([source, target, edge_attr], dim=1)
        # add linear features and skip
        x = self.nn(x) + self.lin(edge_attr)
        # add bilinear features
        #x = x + self.bilin(source, edge_attr) + self.bilin(target, edge_attr)
        
        return self.bn(x)


class NNConvModel(torch.nn.Module):
    """
    GNN with 
    * NNConv on edges
    * MetaLayer on nodes
    
    followed by MetLayer for prediction
    
    for use in config
    model:
        modules:
            edge_model:
              name: nnconv
    """
    def __init__(self, cfg):
        super(NNConvModel, self).__init__()
        
        
        if 'modules' in cfg:
            self.model_config = cfg['modules']['edge_nnconv']
        else:
            self.model_config = cfg
            
            
        self.node_in = self.model_config.get('node_feats', 16)
        self.edge_in = self.model_config.get('edge_feats', 10)
        
        self.aggr = self.model_config.get('aggr', 'add')
        self.leak = self.model_config.get('leak', 0.1)
        self.use_bn = self.model_config.get('batch_norm', False)
        bn_momentum = self.model_config.get('batch_norm_momentum', 0.1)
        
        # perform batch normalization
        self.bn_node = BatchNorm1d(self.node_in, momentum=bn_momentum)
        self.bn_edge = BatchNorm1d(self.edge_in, momentum=bn_momentum)
        
        self.num_mp = self.model_config.get('num_mp', 3)
        nnode_feats = self.node_in
        nedge_feats = self.edge_in
        
        self.nn = torch.nn.ModuleList()
        self.layer = torch.nn.ModuleList()
        self.em = torch.nn.ModuleList()
        self.bn = torch.nn.ModuleList()

        for i in range(self.num_mp):
            einput = nedge_feats
            ninput = nnode_feats
            # maintain this many node features
            noutput = max(2*nnode_feats, 16)
            eoutput = max(2*nedge_feats, 32)
            self.em.append(
                MetaLayer(EdgeModel(ninput, einput, eoutput, self.leak, bn_momentum))
            )
            self.nn.append(
                Seq(
                    Lin(eoutput, eoutput*2),
                    LeakyReLU(self.leak),
                    BatchNorm1d(eoutput*2, momentum=bn_momentum),
                    Lin(eoutput*2, ninput*noutput)
                )
            )
            self.layer.append(
                NNConv(ninput, noutput, self.nn[i], aggr=self.aggr)
            )
            self.bn.append(
                BatchNorm1d(noutput, momentum=bn_momentum)
            )

            nnode_feats = noutput
            nedge_feats = eoutput

        # final prediction layer
        pred_cfg = self.model_config.get('pred_model', 'basic')
        if pred_cfg == 'basic':
            self.predictor = MetaLayer(EdgePredModel(nnode_feats, nedge_feats, self.leak), 
                                       NodePredModel(nnode_feats, nedge_feats, self.leak)
                                      )
        elif pred_cfg == 'bilin':
            self.predictor = MetaLayer(BilinEdgePredModel(nnode_feats, nedge_feats, self.leak), 
                                       NodePredModel(nnode_feats, nedge_feats, self.leak)
                                      )
        elif pred_cfg == 'comb':
            self.predictor = MetaLayer(CombEdgePredModel(nnode_feats, nedge_feats, self.leak, bn_momentum=bn_momentum), 
                                       NodePredModel(nnode_feats, nedge_feats, self.leak, bn_momentum=bn_momentum)
                                      )
        else:
            raise Exception('unrecognized prediction model: ' + pred_cfg)
            
        
    def forward(self, x, edge_index, e, xbatch):
        """
        inputs data:
            x - vertex features
            edge_index - graph edge list
            e - edge features
            xbatch - node batchid
        """
        
        x = x.view(-1,self.node_in)
        e = e.view(-1,self.edge_in)
        if self.edge_in > 1:
            e = self.bn_edge(e)
        if self.node_in > 1:
            x = self.bn_node(x)
        
        # go through layers
        for i in range(self.num_mp):
            # update edge features
            x, e, u = self.em[i](x, edge_index, e, u=None, batch=xbatch)
            # update node features
            x = self.layer[i](x, edge_index, e)
            # batch normalize node
            x = self.bn[i](x)
        
        x, e, u = self.predictor(x, edge_index, e, u=None, batch=xbatch)

        return {'edge_pred':[e], 'node_pred':[x]}
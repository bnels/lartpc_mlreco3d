# GNN on clusters.  No primaries

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import numpy as np
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, Sigmoid, LeakyReLU, Dropout, BatchNorm1d
from torch_geometric.nn import MetaLayer, GATConv
from mlreco.utils.gnn.cluster import get_cluster_batch, get_cluster_label, form_clusters_new
from mlreco.utils.gnn.primary import assign_primaries, analyze_primaries
from mlreco.utils.gnn.network import complete_graph
from mlreco.utils.gnn.compton import filter_compton
from mlreco.utils.gnn.data import cluster_vtx_features, cluster_edge_features, edge_assignment, cluster_edge_distance
from mlreco.utils.gnn.data import node_primary_assignment
from mlreco.utils.gnn.evaluation import secondary_matching_vox_efficiency, secondary_matching_vox_efficiency3
from mlreco.utils.gnn.evaluation import DBSCAN_cluster_metrics2, assign_clusters_UF, primary_id_metrics
from mlreco.utils.groups import process_group_data
from .gnn import gnn_model_construct


class CombModel(torch.nn.Module):
    """
    Driver for combined edge and node prediction, assumed to be with PyTorch GNN model.
    This class mostly acts as a wrapper that will hand the graph data to another model
    
    for use in config
    model:
        modules:
            edge_model:
                name: <name of edge model>
                model_cfg:
                    <dictionary of arguments to pass to model>
                remove_compton: <True/False to remove compton clusters> (default True)
    """
    def __init__(self, cfg):
        super(CombModel, self).__init__()
        
        if 'modules' in cfg:
            self.model_config = cfg['modules']['clust_model']
        else:
            self.model_config = cfg
        
        self.remove_compton = self.model_config.get('remove_compton', True)
        self.compton_thresh = self.model_config.get('compton_thresh', 30)
        
        # decide if only to use MST edges
        self.mst_edges_only = self.model_config.get('mst_edges_only', False)
            
        # extract the model to use
        model = gnn_model_construct(self.model_config.get('name', 'edge_only'))
                     
        # construct the model
        self.predictor = model(self.model_config.get('model_cfg', {}))
        
    def forward(self, data):
        """
        inputs data:
            data[0] - dbscan data
        output:
        dictionary, with
            'edge_pred': torch.tensor with edge prediction weights
        """
        # get device
        device = data[0].device
        
        # need to form graph, then pass through GNN
        clusts = form_clusters_new(data[0])
        
        # remove compton clusters
        # if no cluster fits this condition, return
        if self.remove_compton:
            selection = filter_compton(clusts, self.compton_thresh) # non-compton looking clusters
            if not len(selection):
                e = torch.tensor([], requires_grad=True)
                e.to(device)
                return {'edge_pred':[e]}

            clusts = clusts[selection]
        
        # form graph
        batch = get_cluster_batch(data[0], clusts)
        edge_index = complete_graph(batch, device=device)
        
        if not edge_index.shape[0]:
            e = torch.tensor([], requires_grad=True)
            e.to(device)
            return {'edge_pred':[e]}

        # obtain vertex features
        x = cluster_vtx_features(data[0], clusts, device=device)
        
        # if only using mst edges, do filtering here:
        # print(edge_index.shape)
        if self.mst_edges_only:
            # compute distance-based MST
            de = cluster_edge_distance(data[0], clusts, edge_index, device=device) # edge distance
            # index of MST edges
            mst_inds = get_mst_inds(edge_index, -de, len(clusts))
            # mst edges
            edge_index = edge_index[:,mst_inds]
            # print(edge_index)
        # print(edge_index.shape)
        
        # obtain edge features
        e = cluster_edge_features(data[0], clusts, edge_index, device=device)
        
        # get x batch
        xbatch = torch.tensor(batch).to(device)
        
        # get output
        out = self.predictor(x, edge_index, e, xbatch)
        # print(len(clusts))
        return {
            'clusts': clusts,
            'edge_index': edge_index,
            **out
        }
    
def get_mst_inds(edge_index, edge_wt, n):
    """
    Get edges in MST from edge_index
    """
    from topologylayer.functional.persistence import getEdgesUF_raw
    
    edges = edge_index.detach().cpu().numpy()
    edges = edges.T # transpose
    edges = edges.flatten()
    
    val = edge_wt.detach().cpu().numpy()
    
    mst_edges = getEdgesUF_raw(edges, val, n)
    return mst_edges
    
    
class CombChannelLoss(torch.nn.Module):
    """
    Edge loss based on two channel output (off/on)
    combined with node channel loss (non-primary/primary)
    """
    def __init__(self, cfg):
        # torch.nn.MSELoss(reduction='sum')
        # torch.nn.L1Loss(reduction='sum')
        super(CombChannelLoss, self).__init__()
        self.model_config = cfg['modules']['clust_model']
        
        self.remove_compton = self.model_config.get('remove_compton', True)
        self.compton_thresh = self.model_config.get('compton_thresh', 30)
        
        self.reduction = self.model_config.get('reduction', 'mean')
        self.loss = self.model_config.get('loss', 'CE')
        
        self.node_wt = self.model_config.get('node_wt', 0.5)
        on_wt = self.model_config.get('on_wt', 1.0)
        weight = torch.tensor([1.0, on_wt], dtype=torch.float)
        # parameters for MST
        self.mst_wt = self.model_config.get('mst_wt', 0.0)
        mst_on_wt = self.model_config.get('mst_on_wt', on_wt)
        mst_weight = torch.tensor([1.0, mst_on_wt], dtype=torch.float)
        
        if self.loss == 'CE':
            self.lossfn = torch.nn.CrossEntropyLoss(weight=weight, reduction=self.reduction)
            self.lossmst = torch.nn.CrossEntropyLoss(weight=mst_weight, reduction=self.reduction)
        elif self.loss == 'MM':
            p = self.model_config.get('p', 1)
            margin = self.model_config.get('margin', 1.0)
            self.lossfn = torch.nn.MultiMarginLoss(p=p, margin=margin, reduction=self.reduction)
            self.lossmst = torch.nn.MultiMarginLoss(p=p, margin=margin, reduction=self.reduction)
        else:
            raise Exception('unrecognized loss: ' + self.loss)
        
        
    def forward(self, out, clusters, groups, primary):
        """
        out:
            dictionary output from GNN Model
            keys:
                'edge_index': edges used
                'edge_pred': predicted edge weights from model forward
        data:
            data[0] - DBSCAN data
            data[1] - groups data
        """
        edge_ct = 0
        total_loss, total_acc = 0., 0.
        mst_loss = 0.
        ari, ami, sbd, pur, eff = 0., 0., 0., 0., 0.
        ppur, peff, spur, seff = 0., 0., 0., 0.
        node_loss = 0.
        edge_loss = 0.
        # count edges on/off
        true_on, true_off, est_on, est_off = 0, 0, 0, 0
        # number of true and assigned clusters
        ntrue, nfound = 0, 0
        ngpus = len(clusters)
        for i in range(ngpus):
            edge_pred = out['edge_pred'][i]
            # print(edge_pred.shape)
            node_pred = out['node_pred'][i]
            # print(node_pred.shape)
            data0 = clusters[i]
            data1 = groups[i]
            data2 = primary[i]

            device = data0.device

            # first decide what true edges should be
            # need to form graph, then pass through GNN
            # clusts = form_clusters(data0)
            # clusts = form_clusters_new(data0)
            
            # remove compton clusters
            # if no cluster fits this condition, return
            clusts = out['clusts'] # load cached clusters

            # process group data
            # data_grp = process_group_data(data1, data0)
            data_grp = data1

            # form graph
            batch = get_cluster_batch(data0, clusts)
            # edge_index = complete_graph(batch, device=device)
            edge_index = out['edge_index'] # load cached edge index

            if not edge_index.shape[0]:
                total_loss += self.lossfn(edge_pred, edge_pred)
                totalacc += 1.
                continue
                
            group = get_cluster_label(data_grp, clusts)
            ntrue += len(np.unique(group))

            # determine true assignments
            # print(edge_index.shape)
            # print(len(clusts))
            edge_assn = edge_assignment(edge_index, batch, group, device=device, dtype=torch.long)
            edge_assn = edge_assn.view(-1)
            true_on += torch.sum(edge_assn).detach().cpu().item()
            
            node_assn = node_primary_assignment(data2, clusts, data1, device=device, dtype=torch.long)
            node_assn = node_assn.view(-1)

            # total loss on batch
            edge_loss += self.lossfn(edge_pred, edge_assn)
            node_loss += self.lossfn(node_pred, node_assn)
            
            
            # get fraction of assigned primaries that are correct
            ppur0, peff0, spur0, seff0 = primary_id_metrics(node_pred, node_assn)
            ppur += ppur0
            peff += peff0
            spur += spur0
            seff += seff0

            # compute assigned clusters
            fe = edge_pred[:,1] - edge_pred[:,0]
            cs = assign_clusters_UF(edge_index, fe, len(clusts), thresh=0.0)
            est_on += torch.sum(fe > 0).detach().cpu().item()
            nfound += len(np.unique(cs))
            
            if self.mst_wt > 0:
                # get active edges
                mst_inds = get_mst_inds(edge_index, fe, len(clusts))
                mst_pred = edge_pred[mst_inds]
                mst_assn = edge_assn[mst_inds]
                mst_loss += self.lossmst(mst_pred, mst_assn)

            ari0, ami0, sbd0, pur0, eff0 = DBSCAN_cluster_metrics2(
                cs,
                clusts,
                group
            )
            ari += ari0
            ami += ami0
            sbd += sbd0
            pur += pur0
            eff += eff0

            edge_ct += edge_index.shape[1]
            
        total_loss = (1 - self.node_wt) * ((1 - self.mst_wt) * edge_loss + self.mst_wt * mst_loss) +\
            self.node_wt * node_loss
        
        return {
            'ARI': ari/ngpus,
            'AMI': ami/ngpus,
            'SBD': sbd/ngpus,
            'purity': pur/ngpus,
            'efficiency': eff/ngpus,
            'accuracy': sbd/ngpus, # sbd
            'loss': total_loss/ngpus,
            'node_loss': node_loss/ngpus,
            'edge_loss': edge_loss/ngpus,
            'mst_loss':  mst_loss/ngpus,
            'edge_count': edge_ct,
            'true_on': true_on,
            'est_on': est_on,
            'true_clusts': ntrue,
            'est_clusts': nfound,
            'primary_pur': ppur/ngpus,
            'primary_eff': peff/ngpus,
            'secondary_pur': spur/ngpus,
            'secondary_eff': seff/ngpus
        }

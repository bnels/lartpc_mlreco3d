# utility to evaluate the network accuracy
import numpy as np
import torch
from mlreco.utils.metrics import SBD, AMI, ARI, purity_efficiency
import time


def assign_clusters(edge_index, edge_label, primaries, others, n):
    """
    assigns each node to a cluster represented by the primary node
    """
    clust = np.zeros(n)
    for i in primaries:
        clust[i] = i
    for i in others:
        inds = edge_index[1,:] == i
        if sum(inds) == 0:
            clust[i] = -1
            continue
        indmax = torch.argmax(edge_label[inds])
        clust[i] = edge_index[0,inds][indmax].item()
    return clust


def assign_clusters_UF(edge_index, edge_wt, n, thresh=0.0):
    """
    assigns clusters using Union Find on edges
    """
    from topologylayer.functional.persistence import getClustsUF_raw
    
    edges = edge_index.detach().cpu().numpy()
    edges = edges.T # transpose
    edges = edges.flatten()
    
    val = edge_wt.detach().cpu().numpy()
    
    cs = getClustsUF_raw(edges, val, n, thresh)
    un, cinds = np.unique(cs, return_inverse=True)
    return cinds


def secondary_matching_vox_efficiency(edge_index, true_labels, pred_labels, primaries, clusters, n):
    """
    fraction of secondary voxels that are correctly assigned
    """
    # mask = np.array([(i not in primaries) for i in range(n)])
    # others = np.arange(n)[mask]
    others = np.array([i for i in range(n) if i not in primaries])
    true_nodes = assign_clusters(edge_index, true_labels, primaries, others, n)
    pred_nodes = assign_clusters(edge_index, pred_labels, primaries, others, n)
    tot_vox = np.sum([len(clusters[i]) for i in others])
    int_vox = np.sum([len(clusters[i]) for i in others if true_nodes[i] == pred_nodes[i]])
    return int_vox * 1.0 / tot_vox


def secondary_matching_vox_efficiency2(matched, group, primaries, clusters):
    """
    fraction of secondary voxels that are correctly assigned
    uses matched array
    """
    n = len(matched)
    others = np.array([i for i in range(n) if i not in primaries])
    others_matched = np.array([i for i in others if matched[i] > -1])
    tot_vox = np.sum([len(clusters[i]) for i in others])
    int_vox = np.sum([len(clusters[i]) for i in others_matched if  group[i] == group[matched[i]]])
    return int_vox * 1.0 / tot_vox


def secondary_matching_vox_efficiency3(edge_index, true_labels, pred_labels, primaries, clusters, n):
    """
    fraction of secondary voxels that are correctly assigned
    pred_labels is N x C
    """
    # mask = np.array([(i not in primaries) for i in range(n)])
    # others = np.arange(n)[mask]
    others = np.array([i for i in range(n) if i not in primaries])
    true_nodes = assign_clusters(edge_index, true_labels, primaries, others, n)
    pred_labels = torch.argmax(pred_labels, 1) # get argmax predicted
    pred_nodes = assign_clusters(edge_index, pred_labels, primaries, others, n)
    tot_vox = np.sum([len(clusters[i]) for i in others])
    int_vox = np.sum([len(clusters[i]) for i in others if true_nodes[i] == pred_nodes[i]])
    return int_vox * 1.0 / tot_vox


def primary_assign_vox_efficiency(true_nodes, pred_nodes, clusters):
    """
    fraction of secondary voxels that are correctly assigned
    """
    tot_vox = np.sum([len(c) for c in clusters])
    int_vox = np.sum([len(clusters[i]) for i in range(len(clusters)) if np.sign(true_nodes[i].detach().cpu().numpy()) == np.sign(pred_nodes[i].detach().cpu().numpy())])
    return int_vox * 1.0 / tot_vox


def cluster_to_voxel_label(label, clusters):
    """
    turn an array of labels on clusters to an array of labels on voxels
    """
    nvoxels = np.sum([len(c) for c in clusters])
    vlabel = np.empty(nvoxels, dtype=np.int)
    stptr = 0
    for i, c in enumerate(clusters):
        endptr = stptr + len(c)
        vlabel[stptr:endptr] = label[i]
        stptr = endptr
    return vlabel


def DBSCAN_cluster_metrics(edge_index, true_labels, pred_labels, primaries, clusters, n):
    """
    return ARI, AMI, SBD, purity, efficiency
    of matching
    """
    others = np.array([i for i in range(n) if i not in primaries])
    true_nodes = assign_clusters(edge_index, true_labels, primaries, others, n)
    pred_labels = torch.argmax(pred_labels, 1) # get argmax predicted
    pred_nodes = assign_clusters(edge_index, pred_labels, primaries, others, n)
    pred_vox = cluster_to_voxel_label(pred_nodes, clusters)
    true_vox = cluster_to_voxel_label(true_nodes, clusters)
    ari = ARI(pred_vox, true_vox)
    ami = AMI(pred_vox, true_vox)
    sbd = SBD(pred_vox, true_vox)
    pur, eff = purity_efficiency(pred_vox, true_vox)
    return ari, ami, sbd, pur, eff


def DBSCAN_cluster_metrics2(matched, clusters, group):
    """
    return ARI, AMI, SBD, purity, efficiency
    of matching.  Use matched array
    """
    pred_vox = cluster_to_voxel_label(matched, clusters)
    true_vox = cluster_to_voxel_label(group, clusters)
    #t = time.time()
    ari = ARI(pred_vox, true_vox)
    #print('ARI time = {}'.format(time.time() - t))
    #t = time.time()
    ami = AMI(pred_vox, true_vox)
    #print('AMI time = {}'.format(time.time() - t))
    #t = time.time()
    sbd = SBD(pred_vox, true_vox)
    #print('SBD time = {}'.format(time.time() - t))
    #t = time.time()
    pur, eff = purity_efficiency(pred_vox, true_vox)
    #print('P/E time = {}'.format(time.time() - t))
    #t = time.time()
    return ari, ami, sbd, pur, eff
    
    
def primary_id_metrics(node_pred, node_assn, thresh=0.0):
    """
    return purity and efficiency for primary identification
    thresh is threshold for node_pred to be considered primary
    return:
        primary purity
        primary efficiency
        secondary purity
        secondary efficiency
    """
    if isinstance(node_pred, torch.Tensor):
        node_pred = node_pred.detach().cpu().numpy()
    if isinstance(node_assn, torch.Tensor):
        node_assn = node_assn.detach().cpu().numpy()
        
    node_thresh = (node_pred[:,1] - node_pred[:,0]) > thresh
    
    # primary metrics
    nprimary = sum(node_assn)
    npint = sum(np.logical_and(node_thresh, node_assn))
    npred_primary = sum(node_thresh)
    peff = npint * 1.0 / max(nprimary, 1)
    ppur = npint * 1.0 / max(npred_primary, 1)
    
    # secondary metrics
    nsecondary = len(node_assn) - nprimary
    nsint = sum(np.logical_and(node_thresh == False, node_assn == False))
    npred_secondary = len(node_thresh) - npred_primary
    seff = nsint * 1.0 / max(nsecondary, 1)
    spur = nsint * 1.0 / max(npred_secondary, 1)
    
    return ppur, peff, spur, seff
    
    
    
    
    
    
    
    

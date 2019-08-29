def gnn_model_dict():
    """
    returns dictionary of valid gnn models
    """
    
    from . import attention
    from . import attention2
    from . import edge_only
    from . import edge_node_only
    from . import full_edge_node_only
    from . import nnconv
    from . import nnconv_edge
    from . import econv
    from . import meta
    from . import dir_meta
    
    models = {
        "basic_attention" : attention.BasicAttentionModel,
        "basic_attention2": attention2.BasicAttentionModel,
        "edge_only" : edge_only.EdgeOnlyModel,
        "edge_node_only" : edge_node_only.EdgeNodeOnlyModel,
        "full_edge_node_only" : full_edge_node_only.FullEdgeNodeOnlyModel,
        "nnconv" : nnconv.NNConvModel,
        "nnconv_edge" : nnconv_edge.NNConvModel,
        "econv" : econv.EdgeConvModel,
        "emeta" : meta.EdgeMetaModel,
        "dir_meta" : dir_meta.EdgeMetaModel
    }
    
    return models


def gnn_model_construct(name):
    models = gnn_model_dict()
    if not name in models:
        raise Exception("Unknown GNN model name provided")
    return models[name]


def node_model_dict():
    """
    returns dictionary of valid node models
    """
        
    from . import node_attention
    from . import node_econv
    
    models = {
        "node_attention" : node_attention.NodeAttentionModel,
        "node_econv" : node_econv.NodeEconvModel
    }
    

def node_model_construct(name):
    models = node_model_dict()
    if not name in models:
        raise Exception("Unknown edge model name provided")
    return models[name]
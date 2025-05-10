import torch
from tensordict.nn.distributions import NormalParamExtractor
from torch import nn
from torch_geometric.nn import Linear, Sequential, GCNConv, GraphConv, global_mean_pool, avg_pool, to_hetero
from torch_geometric.data import Data, Batch

class single_node_actor(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.propagation_model = Sequential("x, edge_index, batch", [
            (Linear(27, 64), "x -> x"),
            (GraphConv(64, 64), "x, edge_index -> x"),
            nn.Tanh(),
            (Linear(64, 16), "x -> x"),
            nn.Tanh(),
            NormalParamExtractor(),
        ])

    def forward(self, data):
        #propagation model does not take a list so we unpack the list in a batch as explained in torch_geometric
        if isinstance(data, list):
            batch = Batch.from_data_list(data)
            loc, scale = self.propagation_model(batch.x, batch.edge_index)  
        else:
            loc, scale = self.propagation_model(data.x, data.edge_index)
            loc = loc.t().squeeze(-1)
            scale = scale.t().squeeze(-1)
        return loc, scale

class multinode_actor(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.propagation_model = Sequential("x, edge_index, batch", [
            (Linear(11, 64), "x -> x"),
            (GraphConv(64, 64), "x, edge_index -> x"),
            nn.Tanh(),
            (GraphConv(64, 64), "x, edge_index -> x"),
            nn.Tanh(),
            (Linear(64, 16), "x -> x"),
            nn.Tanh(),
            (global_mean_pool, "x, batch -> x"),
            NormalParamExtractor(),
        ])


    def forward(self, data):
        if isinstance(data, list):
            batch = Batch.from_data_list(data)
            loc, scale = self.propagation_model(batch.x, batch.edge_index, batch.batch)  
        else:
            batch_vec = torch.zeros(data.x.size(0), dtype=torch.long, device=data.x.device)
            loc, scale = self.propagation_model(data.x, data.edge_index, batch_vec)
            loc = loc.t().squeeze(-1)
            scale = scale.t().squeeze(-1)
        return loc, scale
    
class hetero_actor(torch.nn.Module):
    def __init__(self):
        super().__init__()

        node_types = ['torso', 'joint']
        edge_types = [('torso', 'connects', 'joint'), ('joint', 'connects', 'torso'),('joint', 'connects', 'joint')]
        self.propagation_model = to_hetero(Sequential("x, edge_index", [
            (Linear(-1, 64), "x -> x"),
            (GraphConv(64, 64), "x, edge_index -> x"),
            nn.Tanh(),
            (GraphConv(64, 64), "x, edge_index -> x"),
            nn.Tanh(),
            (Linear(64, 16), "x -> x"),
            nn.Tanh(),
            #(global_mean_pool, "x, batch -> x"),
            NormalParamExtractor(),
        ]), metadata=(node_types, edge_types))


    def forward(self, data):
        print(data.keys)
        if isinstance(data, list):
            batch = Batch.from_data_list(data)
            loc, scale = self.propagation_model(batch.x_dict, batch.edge_index_dict)  
        else:
            loc, scale = self.propagation_model(data['torso'].x, data['torso'].edge_index)
            print("loc", loc)
            loc = loc.t().squeeze(-1)
            scale = scale.t().squeeze(-1)
        return loc, scale
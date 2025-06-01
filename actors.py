import torch
from tensordict.nn.distributions import NormalParamExtractor
from torch import nn
from torch_geometric.nn import Linear, Sequential, GCNConv, GraphConv, global_mean_pool, HGTConv, HeteroConv, to_hetero
from torch_geometric.data import Data, Batch

class single_node_actor(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.propagation_model = Sequential("x, edge_index", [
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
        #print((loc, scale).shape)
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
        
        metadata = (['torso', 'joint'], [('torso', 'hip', 'joint'), ('joint', 'hip', 'torso'), ('joint', 'knee', 'joint')])
        self.joint_lin = Linear(2, 11)
        self.torso_lin = Linear(11, 11)
        self.propagation_model = to_hetero(Sequential("x, edge_index", [
            (GraphConv(11, 64), "x, edge_index -> x"),
            nn.Tanh(),
            (GraphConv(64, 64), "x, edge_index -> x"),
            nn.Tanh(),
        ]),
            metadata=metadata,
            aggr='sum',)
        self.output_layer = nn.ModuleList([
            Linear(64, 2) for _ in range(8)
        ])

    def forward(self, data):
        if isinstance(data, list):
            batch = Batch.from_data_list(data)
            x_dict = batch.x_dict
            edge_index_dict = batch.edge_index_dict
            x_dict['joint'] = self.joint_lin(x_dict['joint'])
            x_dict['torso'] = self.torso_lin(x_dict['torso'])
            x_dict = self.propagation_model(x_dict, edge_index_dict)
            x_dict['joint'] = torch.stack([self.output_layer[i % 8](x_dict['joint'][i]) for i in range(80)])
            x_dict['joint'] = x_dict['joint'].view(10, 8, -1)
            loc, scale = NormalParamExtractor()(x_dict['joint'])
            loc = loc.squeeze(-1)
            scale = scale.squeeze(-1)
        else:
            x_dict = data.x_dict
            edge_index_dict = data.edge_index_dict
            x_dict['joint'] = self.joint_lin(x_dict['joint'])
            x_dict['torso'] = self.torso_lin(x_dict['torso'])
            x_dict = self.propagation_model(x_dict, edge_index_dict)
            x_dict['joint'] = torch.stack([self.output_layer[i](x_dict['joint'][i]) for i in range(8)])
            loc, scale = NormalParamExtractor()(x_dict['joint'])
            loc = loc.squeeze(-1)
            scale = scale.squeeze(-1)
        return loc, scale

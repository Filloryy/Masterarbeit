import torch
from tensordict.nn.distributions import NormalParamExtractor
from torch import nn
from torch_geometric.nn import Linear, Sequential, GCNConv, GraphConv, global_mean_pool, HGTConv, HeteroConv, to_hetero
from torch_geometric.data import Data, Batch

def mlp_actor(num_cells=256, action_dim=8, device="cpu"):
    return nn.Sequential(
        nn.LazyLinear(num_cells, device=device),
        nn.Tanh(),
        nn.LazyLinear(num_cells, device=device),
        nn.Tanh(),
        nn.LazyLinear(num_cells, device=device),
        nn.Tanh(),
        nn.LazyLinear(2 * action_dim, device=device),
        NormalParamExtractor(),
    )

class single_node_actor(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.propagation_model = Sequential("x, edge_index", [
            (Linear(27, 64), "x -> x"),
            (GraphConv(64, 64), "x, edge_index -> x"),
            nn.Tanh(),
            (GraphConv(64, 64), "x, edge_index -> x"),
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

class distributed_actor(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.propagation_model = Sequential("x, edge_index, batch", [
            (Linear(11, 64), "x -> x"),
            (GraphConv(64, 64), "x, edge_index -> x"),
            nn.Tanh(),
            (GraphConv(64, 64), "x, edge_index -> x"),
            nn.Tanh(),
            (global_mean_pool, "x, batch -> x"),
            (Linear(64, 16), "x -> x"),
            NormalParamExtractor(),
        ])

        self.num_nodes_per_graph = 9  # Assuming each graph has 9 nodes

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
    
class no_batching_actor(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.propagation_model = Sequential("x, edge_index", [
            (Linear(11, 64), "x -> x"),
            (GraphConv(64, 64), "x, edge_index -> x"),
            nn.Tanh(),
            (GraphConv(64, 64), "x, edge_index -> x"),
            nn.Tanh(),

        ])
        self.output_layer = nn.ModuleList([
            Linear(64, 2) for _ in range(8)
        ])
        self.num_nodes_per_graph = 9  # Assuming each graph has 9 nodes

    def forward(self, data):
        if isinstance(data, list):
            batch = Batch.from_data_list(data)
            batch.x = self.propagation_model(batch.x, batch.edge_index)
            batch.x = torch.stack([self.output_layer[(i % 9) -1](batch.x[i]) for i in range(90) if i % 9 != 0])
            batch.x = batch.x.view(10, 8, -1)
            loc, scale = NormalParamExtractor()(batch.x)
            loc = loc.squeeze(-1)
            scale = scale.squeeze(-1)
        else:
            x = self.propagation_model(data.x, data.edge_index)
            x = torch.stack([self.output_layer[i-1](x[i]) for i in range(9) if i % 9 != 0])
            loc, scale = NormalParamExtractor()(x)
            loc = loc.squeeze(-1)
            scale = scale.squeeze(-1)
        return loc, scale

class left_right_actor(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.propagation_model = Sequential("x, edge_index, batch", [
            (Linear(11, 64), "x -> x"),
            (GraphConv(64, 64), "x, edge_index -> x"),
            nn.Tanh(),
            (GraphConv(64, 64), "x, edge_index -> x"),
            (global_mean_pool, "x, batch -> x"),
            (Linear(64, 16), "x -> x"),
            NormalParamExtractor(),
        ])
        self.output_layer = Linear(64, 2)
        self.num_nodes_per_graph = 3

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
        self.joint_lin = Linear(2, 64)
        self.torso_lin = Linear(11, 64)
        self.propagation_model = to_hetero(Sequential("x, edge_index", [
            (GraphConv(64, 64), "x, edge_index -> x"),
            nn.Tanh(),
            (GraphConv(64, 64), "x, edge_index -> x"),
            nn.Tanh(),
            # (GraphConv(64, 64), "x, edge_index -> x"),
            # nn.Tanh(),
        ]),
            metadata=metadata,
            aggr='sum',)
        self.output_layer = nn.ModuleList([
            Linear(64, 2) for _ in range(8)
        ])
        #self.output_layer = Linear(64, 2)

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
    
class hetero_full_info_actor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        metadata = (['torso', 'joint'], [('torso', 'hip', 'joint'), ('joint', 'hip', 'torso'), ('joint', 'knee', 'joint')])
        self.joint_lin = Linear(36, 128)
        self.torso_lin = Linear(36, 128)
        self.propagation_model = to_hetero(Sequential("x, edge_index", [
            (GraphConv(128, 128), "x, edge_index -> x"),
            nn.Tanh(),
            (GraphConv(128, 128), "x, edge_index -> x"),
            nn.Tanh(),
        ]),
            metadata=metadata,
            aggr='sum',)
        self.output_layer = nn.ModuleList([
            Linear(128, 2) for _ in range(8)
        ])
        #self.output_layer = Linear(64, 2)

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
    
class investigactor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.actiono = [-1., -1., -1., -1., -1., -1., 1., -1.], [0., 0., 0., 0., 0., 0., 0., 0.]

    def forward(self, data):
        if isinstance(data, list):
            loc, scale = [self.actiono, self.actiono, self.actiono, self.actiono, self.actiono, self.actiono, self.actiono, self.actiono, self.actiono, self.actiono]
            print("Investigactor action:", loc, scale)
        else:
            loc, scale = self.actiono
        return loc, scale

class contact_actor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        metadata = (['torso', 'joint'], [('torso', 'hip', 'joint'), ('joint', 'hip', 'torso'), ('joint', 'knee', 'joint')])
        self.joint_lin = Linear(120, 128)
        self.torso_lin = Linear(120, 128)
        self.propagation_model = to_hetero(Sequential("x, edge_index", [
            (GraphConv(128, 128), "x, edge_index -> x"),
            nn.Tanh(),
            (GraphConv(128, 128), "x, edge_index -> x"),
            nn.Tanh(),
        ]),
            metadata=metadata,
            aggr='sum',)
        self.output_layer = nn.ModuleList([
            Linear(128, 2) for _ in range(8)
        ])
        #self.output_layer = Linear(64, 2)

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
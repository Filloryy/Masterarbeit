import torch
from tensordict.nn.distributions import NormalParamExtractor
from torch import nn
from torch_geometric.nn import Linear, Sequential, GCNConv, GraphConv, global_mean_pool, HGTConv, HeteroConv
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
    def __init__(self, hidden_channels=64, out_channels=2, num_heads=2, num_layers=2):
        super().__init__()
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        metadata = (['torso', 'joint'], [('torso', 'connects', 'joint'), ('joint', 'connects', 'torso'), ('joint', 'connects', 'joint')])
        node_types, edge_types = metadata 
        self.joint_lin = Linear(2, 11)
        self.torso_lin = Linear(11, 11)
        self.conv_layer = HeteroConv({          #this is a single layer
            ('torso', 'connects', ' joint'): GraphConv(11, 64),
            ('joint', 'connects', 'torso'): GraphConv(11, 64),
            ('joint', 'connects', 'joint'): GraphConv(11, 64),
            }, aggr='sum')
        self.output_layer = Linear(64, 2)
        self.extractor = NormalParamExtractor()


    def forward(self, data):
        #propagation model does not take a list so we unpack the list in a batch as explained in torch_geometric
        if isinstance(data, list):
            batch = Batch.from_data_list(data)
            batch_dict = {k: v.batch for k, v in batch._node_store_dict.items()}
            batch['torso'].x = Linear(-1, 11)(batch['torso'].x)
            batch['joint'].x = Linear(-1, 11)(batch['joint'].x)
            x_dict = self.conv_layer(batch.x_dict, batch.edge_index_dict)
            batch['joint'].x = x_dict['joint']
            batch['joint'].x = self.output_layer(batch['joint'].x)
            batch['joint'].x = batch['joint'].x.view(10, 8, -1)
            loc, scale = self.extractor(batch['joint'].x)
            loc = loc.squeeze(-1)
            scale = scale.squeeze(-1)
            
        else:
            batch_vec = [0]
            data['joint'].x = self.joint_lin(data['joint'].x)
            data['torso'].x = self.torso_lin(data['torso'].x)
            x_dict = self.conv_layer(data.x_dict, data.edge_index_dict, batch_vec)
            data['joint'].x = x_dict['joint']
            data['joint'].x = self.output_layer(data['joint'].x)
            loc, scale = self.extractor(data['joint'].x)
            loc = loc.squeeze(-1)
            scale = scale.squeeze(-1)
        return loc, scale
    
        """
        batch = False
        if isinstance(data, list):
            data = Batch.from_data_list(data)
            batch = True

        edge_index_dict = data.edge_index_dict
        x_dict = data.x_dict

        x_dict['joint'] = self.joint_lin(x_dict['joint'])
        x_dict['torso'] = self.torso_lin(x_dict['torso'])

        for conv in self.propagation_model:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {k: self.tanh(v) for k, v in x_dict.items()}

        x_dict['joint'] = self.lin_out(x_dict['joint'])
        x_dict['joint'] = self.tanh(x_dict['joint'])
        loc, scale = self.extractor(x_dict['joint'])
        if batch:
            loc = loc.view(10, 8, -1)
            scale = scale.view(10, 8, -1)
        loc, scale = loc.squeeze(-1), scale.squeeze(-1)
        """
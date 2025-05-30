from torchrl.envs.transforms import ObservationTransform
import torch
from torch_geometric.data import Data, HeteroData
from tensordict import NonTensorData, TensorDictBase
from torchrl.data.tensor_specs import Bounded
from torchrl.envs.transforms.transforms import _apply_to_composite
from torch_geometric.utils import dense_to_sparse
from torch_geometric import transforms as T

class ObsToGraph(ObservationTransform):
    def _apply_transform(self, obs: torch.Tensor) -> NonTensorData:
        #just an example for a fully disconnected graph, each observation is a node
        #should be replaced with one node per joint and edges for limbs
        x = torch.tensor([[obs[0], obs[1],],# """obs[2], obs[3], obs[4], obs[13], obs[14], obs[15], obs[16], obs[17]"""],
                            [obs[5], obs[19]],
                            [obs[6], obs[20]],
                            [obs[7], obs[21]],
                            [obs[8], obs[22]],
                            [obs[9], obs[23]],
                            [obs[10], obs[24]],
                            [obs[11], obs[25]],
                            [obs[12], obs[26]],],
                              dtype=torch.float)
             
        edge_index = torch.tensor([[0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 6, 7, 8],
                                   [1, 2, 3, 4, 5, 0, 6, 0, 7, 0, 8, 0, 1, 2, 3, 4]],
                                    dtype=torch.long)
        data = Data(x=x, edge_index=edge_index)
        return data
    
    # The transform must also modify the data at reset time
    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        return self._call(tensordict_reset)

    # _apply_to_composite will execute the observation spec transform across all
    # in_keys/out_keys pairs and write the result in the observation_spec which
    # is of type ``Composite``
    @_apply_to_composite
    def transform_observation_spec(self, observation_spec):
        return Bounded(
            low=-1,
            high=1,
            shape=observation_spec.shape,
            dtype=observation_spec.dtype,
            device=observation_spec.device,
        )
    
class OneNode(ObservationTransform):
    def _apply_transform(self, obs: torch.Tensor) -> NonTensorData:
        #single node for all observations, kinda works like a normal nn
        x = torch.tensor([[obs[0],
                        obs[1],
                        obs[2],
                        obs[3],
                        obs[4],
                        obs[5],
                        obs[6],
                        obs[7],
                        obs[8],
                        obs[9],
                        obs[10],
                        obs[11],
                        obs[12],
                        obs[13],
                        obs[14],
                        obs[15],
                        obs[16],
                        obs[17],
                        obs[18],
                        obs[19],
                        obs[20],
                        obs[21],
                        obs[22],
                        obs[23],
                        obs[24],
                        obs[25],
                        obs[26]]], dtype=torch.float)
        
        edge_index = torch.tensor([[0, 0]], dtype=torch.long).t() #added self loop for propagation
        data = Data(x=x, edge_index=edge_index)
        return data
    
    # The transform must also modify the data at reset time
    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        return self._call(tensordict_reset)

    # _apply_to_composite will execute the observation spec transform across all
    # in_keys/out_keys pairs and write the result in the observation_spec which
    # is of type ``Composite``
    @_apply_to_composite
    def transform_observation_spec(self, observation_spec):
        return Bounded(
            low=-1,
            high=1,
            shape=observation_spec.shape,
            dtype=observation_spec.dtype,
            device=observation_spec.device,
        )

class torsoleftright(ObservationTransform):
    def _apply_transform(self, obs: torch.Tensor) -> NonTensorData:
        #just an example for a fully disconnected graph, each observation is a node
        #should be replaced with one node per joint and edges for limbs
        x = torch.tensor([[obs[0],  #torso node
                        obs[1],
                        obs[2],
                        obs[3],
                        obs[4],
                        obs[13],
                        obs[14],
                        obs[15],
                        obs[16],
                        obs[17],
                        obs[18],],
                        [obs[5],    #left node
                        obs[6],
                        obs[9],
                        obs[10],
                        obs[19],
                        obs[20],
                        obs[23],
                        obs[24],
                        0,
                        0,
                        0,],
                        [obs[7],    #right node
                        obs[8],
                        obs[11],
                        obs[12],
                        obs[21],
                        obs[22],
                        obs[25],
                        obs[26],
                        0,
                        0,
                        0,]],
                        dtype=torch.float)
        
        edge_index = torch.tensor([[0, 0], [0, 1], [1, 0], [0, 2], [2, 0]], dtype=torch.long).t() 
        data = Data(x=x, edge_index=edge_index)
        return data
    
    # The transform must also modify the data at reset time
    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        return self._call(tensordict_reset)

    # _apply_to_composite will execute the observation spec transform across all
    # in_keys/out_keys pairs and write the result in the observation_spec which
    # is of type ``Composite``
    @_apply_to_composite
    def transform_observation_spec(self, observation_spec):
        return Bounded(
            low=-1,
            high=1,
            shape=observation_spec.shape,
            dtype=observation_spec.dtype,
            device=observation_spec.device,
        )

class fullbodygraph(ObservationTransform):
    def _apply_transform(self, obs: torch.Tensor) -> NonTensorData:
        x = torch.tensor([[obs[0], obs[1], obs[2], obs[3], obs[4], obs[13], obs[14], obs[15], obs[16], obs[17], obs[18]],
                            [obs[5], obs[19], 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [obs[6], obs[20], 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [obs[7], obs[21], 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [obs[8], obs[22], 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [obs[9], obs[23], 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [obs[10], obs[24], 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [obs[11], obs[25], 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [obs[12], obs[26], 0, 0, 0, 0, 0, 0, 0, 0, 0],],
                              dtype=torch.float)     
        
        edge_index = torch.tensor([[0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 6, 7, 8],
                                   [1, 2, 3, 4, 5, 0, 6, 0, 7, 0, 8, 0, 1, 2, 3, 4]],
                                    dtype=torch.long)
        data = Data(x=x, edge_index=edge_index)
        return data
    
    # The transform must also modify the data at reset time
    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        return self._call(tensordict_reset)

    # _apply_to_composite will execute the observation spec transform across all
    # in_keys/out_keys pairs and write the result in the observation_spec which
    # is of type ``Composite``
    @_apply_to_composite
    def transform_observation_spec(self, observation_spec):
        return Bounded(
            low=-1,
            high=1,
            shape=observation_spec.shape,
            dtype=observation_spec.dtype,
            device=observation_spec.device,
        )
    
class heterograph(ObservationTransform):
    def _apply_transform(self, obs: torch.Tensor) -> NonTensorData:
        data = HeteroData()
        data['torso'].x = torch.tensor([[obs[0], obs[1], obs[2], obs[3], obs[4], obs[13], obs[14], obs[15], obs[16], obs[17], obs[18]],
                                ], dtype=torch.float)
        
        data['joint'].x = torch.tensor([[obs[11], obs[25]],
                                        [obs[12], obs[26]],
                                        [obs[7], obs[21]],
                                        [obs[8], obs[22]],
                                        [obs[9], obs[23]],
                                        [obs[10], obs[24]],
                                        [obs[5], obs[19]],
                                        [obs[6], obs[20]],
                                        ], dtype=torch.float)
        
        data['torso', 'hip', 'joint'].edge_index = torch.tensor([[0, 0, 0, 0], [0, 2, 4, 6]], dtype=torch.long)
        data['joint', 'hip', 'torso'].edge_index = torch.tensor([[0, 2, 4, 6], [0, 0, 0, 0]], dtype=torch.long)
        data['joint', 'knee', 'joint'].edge_index = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7], [1, 0, 3, 2, 5, 4, 7, 6]], dtype=torch.long)
        data['torso', 'connects', 'torso'].edge_index = torch.tensor(([0], [0]), dtype=torch.long)
        return data
    
    #The transform must also modify the data at reset time
    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        return self._call(tensordict_reset)

    # _apply_to_composite will execute the observation spec transform across all
    # in_keys/out_keys pairs and write the result in the observation_spec which
    # is of type ``Composite``
    @_apply_to_composite
    def transform_observation_spec(self, observation_spec):
        return Bounded(
            low=-1,
            high=1,
            shape=observation_spec.shape,
            dtype=observation_spec.dtype,
            device=observation_spec.device,
        )
    
class Notransform(ObservationTransform):
    def _apply_transform(self, obs: torch.Tensor) -> NonTensorData:
        return obs
    

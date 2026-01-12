from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium
import torch
from typing import Dict, Callable
from utils.dimensionality_reduction import four_band_reduction, k_local_reduction_4k

class CustomFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gymnasium.spaces.Dict, features_dim : int ):
        super().__init__(observation_space=observation_space,features_dim=features_dim)
        self._extractors: Dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = {}
    
    @property
    def extractors(self):
        return self._extractors
    
    @extractors.setter
    def extractors(self, value):
        self._extractors = value

    def forward(self, observations: Dict[str,torch.Tensor]) -> torch.Tensor:
        if not len(self.extractors):
            raise NotImplementedError()
        encoded_tensor_list = []
        num_agents = observations.pop("num_agents").long().squeeze(-1)
        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            extracted = extractor(observations[key], num_agents)
            encoded_tensor_list.append(extracted)
        return torch.cat(encoded_tensor_list,dim=1)

class _4BandFeatureExtractor(CustomFeatureExtractor):
    def __init__(self, observation_space: gymnasium.spaces.Dict, features_dim : int ):
        super().__init__(observation_space=observation_space,features_dim=features_dim)
        self.four_band_reducer = lambda configs, num_agents: four_band_reduction(configs,num_agents)

class K_Local_FeatureExtractor(CustomFeatureExtractor):
    def __init__(self, observation_space: gymnasium.spaces.Dict, k : int, features_dim : int ):
        super().__init__(observation_space=observation_space,features_dim=features_dim)
        self.k_local_reducer = lambda configs, num_agents: k_local_reduction_4k(configs,num_agents,k)


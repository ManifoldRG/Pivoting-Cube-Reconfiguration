from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium
import torch

class CustomFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gymnasium.spaces.Dict, features_dim : int ):
        super().__init__(observation_space=observation_space,features_dim=features_dim)
    
    def forward(self, observations) -> torch.Tensor:
        raise NotImplementedError()


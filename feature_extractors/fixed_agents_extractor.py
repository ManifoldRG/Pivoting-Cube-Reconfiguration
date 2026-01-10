import gymnasium
import torch
import torch.nn as nn
from feature_extractors.base import CustomFeatureExtractor
from utils.dimensionality_reduction import four_band_reduction

# Class implemented for legacy purposes (to train a fixed number of agents)
class FixedAgentsFeatureExtractor(CustomFeatureExtractor):
    def __init__(self, observation_space: gymnasium.spaces.Dict, num_agents: int):
        super().__init__(observation_space=observation_space,features_dim=2*(4*num_agents - 10) + num_agents)
        self.num_agents = num_agents

        self.reducer = lambda configs, num_agents: four_band_reduction(configs,num_agents)[:, : 4*self.num_agents - 10]
        self.extractors = {
            "current_config": self.reducer,
            "final_config": self.reducer,
            "agent_id": lambda agent_ids, _: torch.nn.functional.one_hot(agent_ids.squeeze(-1).long() - 1, num_classes=self.num_agents).float()
        }

    def forward(self, observations) -> torch.Tensor:
        encoded_tensor_list = []
        num_agents = observations.pop("num_agents").long().squeeze(-1)
        for key, extractor in self.extractors.items():
            extracted = extractor(observations[key], num_agents)
            encoded_tensor_list.append(extracted)
        feature_vector = torch.cat(encoded_tensor_list,dim=1)
        return feature_vector


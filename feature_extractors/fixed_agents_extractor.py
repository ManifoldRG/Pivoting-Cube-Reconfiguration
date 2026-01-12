import gymnasium
import torch
from feature_extractors.base import _4BandFeatureExtractor

# Class implemented for legacy purposes (to train a fixed number of agents)
class FixedAgentsFeatureExtractor(_4BandFeatureExtractor):
    def __init__(self, observation_space: gymnasium.spaces.Dict, num_agents: int):
        super().__init__(observation_space=observation_space,features_dim=2*(4*num_agents - 10) + num_agents)
        self.num_agents = num_agents

        self.extractors = {
            "current_config": self.four_band_reducer,
            "final_config": self.four_band_reducer,
            "agent_id": lambda agent_ids, _: torch.nn.functional.one_hot(agent_ids.squeeze(-1).long() - 1, num_classes=self.num_agents).float()
        }
    


import gymnasium
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from feature_extractors.base import CustomFeatureExtractor
from utils.positional_embeddings import sinusoidal_embedding

class RecurrentFeatureExtractor(CustomFeatureExtractor):
    def __init__(self, observation_space: gymnasium.spaces.Dict, embedded_dim = 128):
        self.agent_id_dim = 16  # Dimension for sinusoidal embedding
        self.embedded_dim = embedded_dim
        super().__init__(observation_space=observation_space,features_dim=2*self.embedded_dim + self.agent_id_dim)

        self.encoder_model = nn.GRU(input_size=4,hidden_size=self.embedded_dim,dtype=torch.float32, batch_first=True)

        self.encoder = lambda observation,num_agents : self.encoder_model(pack_padded_sequence(observation, num_agents, batch_first=True, enforce_sorted=False))[1].squeeze(0)
        self.extractors = {
            "current_config": self.encoder,
            "final_config": self.encoder,
            "agent_id": lambda agent_ids,num_agents: sinusoidal_embedding(self.agent_id_dim, agent_ids)
        }

    def forward(self, observations) -> torch.Tensor:
        encoded_tensor_list = []
        num_agents = observations.pop("num_agents").long().squeeze(-1)
        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            extracted = extractor(observations[key], num_agents)
            encoded_tensor_list.append(extracted)
        return torch.cat(encoded_tensor_list,dim=1)


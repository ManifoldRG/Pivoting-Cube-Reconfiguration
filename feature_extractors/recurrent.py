import gymnasium
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from feature_extractors.base import CustomFeatureExtractor
from utils.positional_embeddings import sinusoidal_embedding

class RecurrentFeatureExtractor(CustomFeatureExtractor):
    def __init__(self, observation_space: gymnasium.spaces.Dict, embedded_dim = 256, agent_id_dim = 64):
        self.agent_id_dim = agent_id_dim  # Dimension for sinusoidal embedding
        self.embedded_dim = embedded_dim
        super().__init__(observation_space=observation_space,features_dim=2*self.embedded_dim + self.agent_id_dim)

        # self.encoder_model = nn.GRU(input_size=4,hidden_size=self.embedded_dim,dtype=torch.float32, batch_first=True)

        # self.encoder = lambda observation,num_agents : self.encoder_model(pack_padded_sequence(observation, num_agents, batch_first=True, enforce_sorted=False))[1].squeeze(0)
        # self.extractors = {
        #     "current_config": self.encoder,
        #     "final_config": self.encoder,
        #     "agent_id": lambda agent_ids,num_agents: sinusoidal_embedding(self.agent_id_dim, agent_ids)
        # }



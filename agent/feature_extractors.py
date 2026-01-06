from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import math

class RecurrentFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gymnasium.spaces.Dict, embedded_dim = 128):
        super().__init__(observation_space=observation_space,features_dim=embedded_dim)
        self.embedded_dim = embedded_dim
        self.agent_id_dim = 16  # Dimension for sinusoidal embedding
        self.encoder_model = nn.GRU(input_size=4,hidden_size=self.embedded_dim,dtype=torch.float32, batch_first=True)
        self._features_dim = 2*self.embedded_dim + self.agent_id_dim

        self.encoder = lambda observation,num_agents : self.encoder_model(pack_padded_sequence(observation, num_agents, batch_first=True, enforce_sorted=False))[1].squeeze(0)
        self.extractors = {
            "current_config": self.encoder,
            "final_config": self.encoder,
            "agent_id": self.sinusoidal_embedding
        }
    
    def sinusoidal_embedding(self, agent_id, num_agents):
        """
        Create sinusoidal positional embeddings for agent IDs.
        
        Args:
            agent_id: Tensor of shape (batch_size, 1) with agent IDs
            num_agents: Not used, kept for interface compatibility
            
        Returns:
            Tensor of shape (batch_size, agent_id_dim)
        """
        batch_size = agent_id.shape[0]
        agent_id = agent_id.squeeze(-1)  # (batch_size,)
        
        # Create embedding matrix
        embedding = torch.zeros(batch_size, self.agent_id_dim, device=agent_id.device, dtype=torch.float32)
        
        # Compute sinusoidal embeddings
        position = agent_id.unsqueeze(1)  # (batch_size, 1)
        div_term = torch.exp(torch.arange(0, self.agent_id_dim, 2, device=agent_id.device, dtype=torch.float32) * 
                            (-math.log(10000.0) / self.agent_id_dim))
        
        embedding[:, 0::2] = torch.sin(position * div_term)
        embedding[:, 1::2] = torch.cos(position * div_term)
        
        return embedding
    def forward(self, observations) -> torch.Tensor:
        encoded_tensor_list = []
        num_agents = observations.pop("num_agents").long().squeeze(-1)
        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            extracted = extractor(observations[key], num_agents)
            encoded_tensor_list.append(extracted)
        return torch.cat(encoded_tensor_list,dim=1)


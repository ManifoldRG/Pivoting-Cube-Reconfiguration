import torch
import math

def sinusoidal_embedding(agent_id_dim, agent_id):
    """
    Create sinusoidal positional embeddings for agent IDs.
    
    Args:
        agent_id: Tensor of shape (batch_size, 1) with agent IDs
        
    Returns:
        Tensor of shape (batch_size, agent_id_dim)
    """
    batch_size = agent_id.shape[0]
    agent_id = agent_id.squeeze(-1)  # (batch_size,)
    
    # Create embedding matrix
    embedding = torch.zeros(batch_size, agent_id_dim, device=agent_id.device, dtype=torch.float32)
    
    # Compute sinusoidal embeddings
    position = agent_id.unsqueeze(1)  # (batch_size, 1)
    div_term = torch.exp(torch.arange(0, agent_id_dim, 2, device=agent_id.device, dtype=torch.float32) * 
                        (-math.log(10000.0) / agent_id_dim))
    
    embedding[:, 0::2] = torch.sin(position * div_term)
    embedding[:, 1::2] = torch.cos(position * div_term)
    
    return embedding
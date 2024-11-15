import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """
    Residual block for the value encoding of the transformer model.
    Following: https://openreview.net/pdf?id=pCbC3aQB5W
    """
    def __init__(self, d_model, hidden_dim, dropout=0.1):
        super(ResidualBlock, self).__init__()
        
        # MLP with one hidden layer
        self.linear1 = nn.Linear(1, hidden_dim)
        #self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, d_model)
        
        # Skip connection (fully linear)
        self.skip_connection = nn.Linear(1, d_model)
        
        # Dropout on the output linear layer
        self.dropout = nn.Dropout(dropout)
        
        # Layer normalization at the output
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(self, x):
        # Pass through the first linear layer
        #hidden = self.relu(self.linear1(x))
        hidden = self.linear1(x)
        
        # Pass through the second linear layer and apply dropout
        output = self.dropout(self.linear2(hidden))
        
        # Apply skip connection and layer normalization
        output = self.layer_norm(output + self.skip_connection(x))
        
        return output
    

"""
class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, dropout_p, max_len):
        super().__init__()
        # Modified version from: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
        # max_len determines how far the position can have an effect on a token (window)
        
        # Info
        self.dropout = nn.Dropout(dropout_p)
        
        # Encoding - From formula
        pos_encoding = torch.zeros(max_len, dim_model)
        positions_list = torch.arange(0, max_len, dtype=torch.float).view(-1, 1) # 0, 1, 2, 3, 4, 5
        division_term = torch.exp(torch.arange(0, dim_model, 2).float() * (-np.log(10000.0)) / dim_model) # 1000^(2i/dim_model)
        
        # PE(pos, 2i) = sin(pos/1000^(2i/dim_model))
        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)
        
        # PE(pos, 2i + 1) = cos(pos/1000^(2i/dim_model))
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)
        
        # Saving buffer (same as parameter without gradients needed)
        pos_encoding = pos_encoding.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pos_encoding",pos_encoding)
        
    def forward(self, token_embedding: torch.tensor) -> torch.tensor:
        # Residual connection + pos encoding
        return self.dropout(token_embedding + self.pos_encoding[:token_embedding.size(0), :])
"""

class TimeSeriesTransformer(nn.Module):
    """
    Transformer model for time series forecasting.
    """
    def __init__(self, d_model=128, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.1, max_len=128):
        super(TimeSeriesTransformer, self).__init__()
        
        # Residual block for value encoding
        #self.value_linear = nn.Linear(1, d_model)
        self.residual_block = ResidualBlock(d_model, dim_feedforward, dropout)
        
        # Sinusoidal positional encoding for time encoding
        self.time_encoding = self.get_sinusoidal_encoding(max_len, d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output layer for predictions
        self.output_layer = nn.Linear(d_model, 1)
    
    def get_sinusoidal_encoding(self, max_len, d_model):
        """
        Create sinusoidal positional encodings for the time steps.
        """
        encoding = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).view(-1, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(np.log(10000.0) / d_model))
        
        encoding[:, 0::2] = torch.sin(pos * div_term)  # Apply sin to even indices
        encoding[:, 1::2] = torch.cos(pos * div_term)  # Apply cos to odd indices
        return encoding.unsqueeze(0)  # Add batch dimension
    
    def forward(self, values, times):
        batch_size, L = times.size()
        # Value embedding with residual block
        # shape: [batch, L, d_model]
        value_embedded = self.residual_block(values.unsqueeze(-1))  # Pass through residual block
        
        # Time embedding using sinusoidal encoding
        time_embedded = self.time_encoding[:, :L, :].repeat(batch_size, 1, 1)  # Match batch and sequence length

        # Normalize embeddings
        value_embedded = F.normalize(value_embedded, p=2, dim=-1)
        time_embedded = F.normalize(time_embedded, p=2, dim=-1)
        
        # Combine time and value embeddings
        x = value_embedded + time_embedded

        # Pass through transformer encoder
        x = self.transformer_encoder(x.permute(1, 0, 2))  # Transformer expects [L, batch, d_model]
        
        # Output prediction for each time step
        output = self.output_layer(x).squeeze(-1)  # shape: [L, batch]
        
        return output.permute(1, 0)  # Return to shape [batch, L]
    

class TrunkNetwork(nn.Module):
    """
    Trunk network that takes the fine grid points as input and outputs a feature vector.
    """
    def __init__(self, dim_model=64, hidden_dim=128, p=128):
        super(TrunkNetwork, self).__init__()
        self.fc1 = nn.Linear(1, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim_model)
        self.fc3 = nn.Linear(dim_model, p)
        #self.activation = nn.ReLU()
        
    def forward(self, y):
        # y is expected to be of shape [batch, num_points, 1] (fine grid points)
        x = self.fc1(y)
        x = self.fc2(x)
        x = self.fc3(x)  # Output shape: [batch, num_points, p]
        return x
    
class DeepONet(nn.Module):
    """
    Combines the TimeSeriesTransformer as the branch network and the trunk Network.
    """
    def __init__(self, X):
        super(DeepONet, self).__init__()
        self.branch = TimeSeriesTransformer()
        self.trunk = TrunkNetwork()

        fine_grid_points = torch.tensor(X,dtype=torch.float32)
        self.fine_grid_points_batch = torch.stack([fine_grid_points for _ in range(64)], dim=0)
    
    def forward(self, values, times):
        # Pass through the branch network (Transformer) to get [b1, ..., bp]
        branch_output = self.branch(values, times)
        
        # Pass through the trunk network to get [t1, ..., tp] for each target point
        trunk_output = self.trunk(self.fine_grid_points_batch.unsqueeze(-1))
        
        # Perform element-wise multiplication and summation to approximate G(u)(y)
        # Shape of trunk_output: [batch, num_points, p]
        # Shape of branch_output: [batch, p]
        # Unsqueeze branch_output to [batch, 1, p] for broadcasting
        output = torch.sum(trunk_output * branch_output.unsqueeze(-1), dim=-1)  # Shape: [batch, num_points]
        
        return output
    
class DeepONetJan(torch.nn.Module):
    """
    Combines Branch and Trunk network with Transformer Encoder for the Branch network.
    """
    def __init__(self, indicator_dim, d_model, heads=1, p=64):
        super(DeepONetJan, self).__init__()
        self.indicator_dim = indicator_dim
        self.p = p
        self.branch_embedding_y = nn.Linear(1, d_model)
        self.branch_embedding_t = nn.Linear(1, d_model)
        self.trunk_embedding_t = nn.Linear(1, d_model)

        self.embedding_act = nn.Sequential(nn.LayerNorm(d_model),nn.LeakyReLU())
        self.branch_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=d_model, nhead=heads, batch_first=True), num_layers=6, enable_nested_tensor=False)

        self.branch_mlp = nn.Sequential(
                            nn.Linear(indicator_dim*d_model,d_model),
                            nn.LeakyReLU(),
                            nn.Linear(d_model, self.p),
                            nn.LeakyReLU()
        )
        self.trunk_mlp =  nn.Sequential(
                            nn.Linear(d_model, d_model),nn.LeakyReLU(),
                            nn.LayerNorm(d_model),
                            nn.Linear(d_model, d_model),nn.LeakyReLU(),
                            nn.Linear(d_model, d_model),nn.LeakyReLU(),
                            nn.LayerNorm(d_model),
                            nn.Linear(d_model, self.p),nn.LeakyReLU()
        )
        
    def forward(self, y, t):

        # Generate the fine grid points batch dynamically for the current batch size
        batch_size = y.shape[0]
        fine_grid_points_batch = torch.linspace(0, 1, self.indicator_dim, device=y.device).unsqueeze(0).expand(batch_size, -1)

        y = y.unsqueeze(-1)
        t = t.unsqueeze(-1)
        t_sample =  fine_grid_points_batch.unsqueeze(-1)

        branch_embedding_y = self.branch_embedding_y(y)
        branch_embedding_t = self.branch_embedding_t(t)
        trunk_encoder_input = self.trunk_embedding_t(t_sample)

        branch_encoder_input = self.embedding_act(branch_embedding_y + branch_embedding_t)
        branch_encoder_output = self.branch_encoder(branch_encoder_input)

        branch_encoder_output = (branch_encoder_output).view(branch_encoder_output.shape[0],-1)
        branch_output = self.branch_mlp(branch_encoder_output) 
        trunk_output = self.trunk_mlp(trunk_encoder_input)

        combined = torch.bmm(branch_output.unsqueeze(1), trunk_output.transpose(1, 2)).squeeze()
        
        return combined
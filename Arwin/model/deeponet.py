import torch
import torch.nn as nn

class DeepONet(torch.nn.Module):
    """
    Combines Branch and Trunk network with Transformer Encoder for the Branch network.
    """
    def __init__(self, indicator_dim, d_model, heads=2, p=128):
        super(DeepONet, self).__init__()
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
        
    def forward(self, y, t, eval_grid_points, mask):
        """ TODO: Do somethin about the fixed grid points"""
        # Generate the fine grid points batch dynamically for the current batch size
        batch_size = y.shape[0]
        fine_grid_points_batch = eval_grid_points.unsqueeze(0).expand(batch_size, -1)

        # Mask the input data
        y = y.unsqueeze(-1) * mask.unsqueeze(-1)
        t = t.unsqueeze(-1) * mask.unsqueeze(-1)
        t_sample =  fine_grid_points_batch.unsqueeze(-1)
        # Branch and Trunk Embedding
        branch_embedding_y = self.branch_embedding_y(y)
        branch_embedding_t = self.branch_embedding_t(t)
        trunk_encoder_input = self.trunk_embedding_t(t_sample)

        # generate mask for the transformer encoder
        mask_enc = torch.where(mask == 1, False, True)

        # Transformer Encoder for the Branch Network
        branch_encoder_input = self.embedding_act(branch_embedding_y + branch_embedding_t)
        branch_encoder_output = self.branch_encoder(branch_encoder_input, src_key_padding_mask=mask_enc)

        # Mask the output of the transformer encoder
        branch_encoder_output = branch_encoder_output * mask.unsqueeze(-1)

        # MLP for the Branch and Trunk Network
        """ TODO: Change flattening """
        branch_encoder_output = (branch_encoder_output).view(branch_encoder_output.shape[0],-1)
        branch_output = self.branch_mlp(branch_encoder_output) 
        trunk_output = self.trunk_mlp(trunk_encoder_input)

        # Combine the Branch and Trunk Network
        combined = torch.bmm(branch_output.unsqueeze(1), trunk_output.transpose(1, 2)).squeeze()
        
        return combined
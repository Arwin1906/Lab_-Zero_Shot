import torch
import torch.nn as nn

class EmbeddingForcaster(torch.nn.Module):
    def __init__(self, d_model, heads=8):
        super(EmbeddingForcaster, self).__init__()
        self.d_model = d_model

        self.scales_mlp = nn.Linear(9 ,d_model)

        # Linear projection layer to adjust output dimension
        self.projection_layer = nn.Sequential(
            nn.Linear(8*d_model, 4*d_model),
            nn.LeakyReLU(),
            nn.Linear(4*d_model, 2*d_model),
            nn.LeakyReLU(),
            nn.Linear(2*d_model, d_model),
            nn.LeakyReLU()        
            )

        self.forecast_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=8*d_model, nhead=heads, batch_first=True), num_layers=4, enable_nested_tensor=False
        )
        self.query = nn.Parameter(torch.randn(1, d_model))
        self.summary_attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=heads, batch_first=True)

    def forward(self, h_b, scales):

        scales_out = self.scales_mlp(scales) # [batch_size*(windows-1), d_model]
        concat = torch.cat((h_b, scales_out), dim=-1) # [batch_size*(windows-1), 2*d_model]

        # Concatenate every 4 tensors
        batch_size, dim = concat.size()
        new_shape = (batch_size // 4, 4 * dim)
        concat_window = concat.view(new_shape) # [batch_size//4, 8*d_model]

        #concat = self.projection_layer(concat) # [batch_size*(windows-1), d_model]

        batch_size = h_b.shape[0]//4
        forecast_out = self.forecast_encoder(concat_window) # [batch_size//4, 8*d_model]
        forecast_out = forecast_out.view(batch_size, -1, self.d_model) # [batch_size, 2*(windows-1), d_model]

        # Attention-based summary
        q = self.query.unsqueeze(0).expand(batch_size, -1, -1)  # Shape: [batch_size, 1, d_model]
        u_b, _ = self.summary_attention(q, forecast_out, forecast_out)  # u_b: [batch_size, 1, d_model]
        #u_b = self.projection_layer(u_b.squeeze(1))  # Flatten to [batch_size, d_model]
        u_b = u_b.squeeze(1)  # Flatten to [batch_size, d_model]

        return u_b
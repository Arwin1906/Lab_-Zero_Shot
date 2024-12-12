import torch
import torch.nn as nn

class EmbeddingForcaster(torch.nn.Module):
    def __init__(self, d_model, heads=8):
        super(EmbeddingForcaster, self).__init__()
        self.d_model = d_model

        self.scales_mlp = nn.Sequential(
                nn.Linear(6 ,d_model),
                nn.LeakyReLU(),
                nn.Linear(d_model, d_model),
                nn.LeakyReLU(),
                nn.Linear(d_model, d_model),
                nn.LeakyReLU(),
        )
        self.forecast_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model*2, nhead=heads, batch_first=True), num_layers=4, enable_nested_tensor=False
        )
        self.query = nn.Parameter(torch.randn(1, 2*d_model))
        self.summary_attention = nn.MultiheadAttention(embed_dim=2*d_model, num_heads=heads, batch_first=True)
        
        # Linear projection layer to adjust output dimension
        self.projection_layer = nn.Linear(2 * d_model, d_model)

    def forward(self, h_b, scales):
        
        batch_size = h_b.shape[0]//4

        scales_out = self.scales_mlp(scales) # [batch_size*(windows-1), d_model]
        concat = torch.cat((h_b, scales_out), dim=-1) # [batch_size*(windows-1), 2*d_model]
        
         # mask should not be required since it was already applied to get h_b and scales do not require masking
        forecast_out = self.forecast_encoder(concat) # [batch_size*(windows-1), 2*d_model]
        forecast_out = forecast_out.view(batch_size, -1, self.d_model*2) # [batch_size, windows-1, 2*d_model]

        # Attention-based summary
        q = self.query.unsqueeze(0).expand(batch_size, -1, -1)  # Shape: [batch_size, 1, 2*d_model]
        u_b, _ = self.summary_attention(q, forecast_out, forecast_out)  # u_b: [batch_size, 1, 2*d_model]
        u_b = self.projection_layer(u_b.squeeze(1))  # Flatten to [batch_size, d_model]

        return u_b
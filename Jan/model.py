# Decompiled with PyLingual (https://pylingual.io)
# Internal filename: /home/jan/Uni/ma-lab/model.py
# Bytecode version: 3.10.0rc2 (3439)
# Source timestamp: 2024-11-11 15:02:10 UTC (1731337330)

import torch
import torch.nn as nn

class DeepONet(torch.nn.Module):

    def __init__(self, indicator_dim, temporal_dim, d_model, heads=1, p=64):
        super(DeepONet, self).__init__()
        self.p = p
        self.branch_embedding_y = nn.Linear(1, d_model)
        self.branch_embedding_t = nn.Linear(1, d_model)
        self.trunk_embedding_t = nn.Linear(1, d_model)
        self.branch_mlp = nn.Sequential(
                            nn.LayerNorm(),
                            nn.Linear(d_model, self.p),
                            nn.Linear(self.p, self.p)
        )
        self.trunk_mlp =  nn.Sequential(
                          
                            nn.Linear(d_model, self.p),
                            nn.Linear(self.p, self.p)
        )
      

        self.branch_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=d_model, nhead=heads, batch_first=True), num_layers=4, enable_nested_tensor=False)
        self.trunk_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=d_model, nhead=heads, batch_first=True), num_layers=4, enable_nested_tensor=False)

    def forward(self, y, t, t_sample, y_mask, t_sample_mask):
        y = y.unsqueeze(-1) * y_mask.unsqueeze(-1)
        t = t.unsqueeze(-1) * y_mask.unsqueeze(-1)
        t_sample = t_sample.unsqueeze(-1) * t_sample_mask.unsqueeze(-1)

        branch_embedding_y = self.branch_embedding_y(y)
        branch_embedding_t = self.branch_embedding_t(t)
        trunk_encoder_input = self.trunk_embedding_t(t_sample)

        y_mask_enc = torch.where(y_mask == 1, False, True)
        t_sample_mask_enc = torch.where(t_sample_mask == 1, False, True)


        branch_encoder_input = branch_embedding_y + branch_embedding_t
        branch_encoder_output = self.branch_encoder(branch_encoder_input, src_key_padding_mask=y_mask_enc)
        trunk_encoder_output = self.trunk_encoder(trunk_encoder_input, src_key_padding_mask=t_sample_mask_enc)

        branch_encoder_output = branch_encoder_output * y_mask.unsqueeze(-1)
        trunk_encoder_output = trunk_encoder_output * t_sample_mask.unsqueeze(-1)

        branch_encoder_output = branch_encoder_output.sum(dim=1)
        print(branch_encoder_output[0])
        print(branch_encoder_output.shape)
        branch_output = self.branch_mlp(branch_encoder_output) 
        trunk_output = self.trunk_mlp(trunk_encoder_output) * t_sample_mask.unsqueeze(-1)

        combined = torch.bmm(branch_output.unsqueeze(1), trunk_output.transpose(1, 2)).squeeze()
        return combined
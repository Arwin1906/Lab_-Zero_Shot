

import torch
import torch.nn as nn

class DeepONet(torch.nn.Module):

    def __init__(self, d_model, heads=1):
        super(DeepONet, self).__init__()
        self.branch_embedding_y = nn.Sequential(nn.Linear(1, d_model))
      
        self.embedding_t_branch =   nn.Sequential(nn.Linear(1, d_model))
        self.embedding_t_trunk =   nn.Sequential(nn.Linear(1, d_model), nn.LayerNorm(d_model))

        self.embedding_act = nn.Sequential(nn.LayerNorm(d_model),nn.LeakyReLU())
        self.branch_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=2*d_model, nhead=heads, batch_first=True), num_layers=6, enable_nested_tensor=False)
     
        
        self.branch_attention = nn.MultiheadAttention(2*d_model, 1 ,batch_first=True)

        self.learnable_q = nn.Parameter(torch.randn(1,2*d_model))

        self.trunk_mlp =   nn.Sequential(nn.Linear(d_model, d_model),nn.LeakyReLU(),
                                         nn.LayerNorm(d_model),
                                         nn.Linear(d_model, d_model),nn.LeakyReLU(),
                                         nn.Linear(d_model, d_model),nn.LeakyReLU(),
                                         nn.LayerNorm(d_model),
                                         nn.Linear(d_model, d_model),nn.LeakyReLU())  
        
        self.final_proj = nn.Sequential(nn.Linear(3*d_model, d_model),nn.LeakyReLU(),
                                         nn.LayerNorm(d_model),
                                         nn.Linear(d_model, d_model),nn.LeakyReLU(),
                                         nn.Linear(d_model, d_model),nn.LeakyReLU(),
                                         nn.LayerNorm(d_model),
                                         nn.Linear(d_model, 1),)
        
    def forward(self, y, t, t_sample, y_mask):
        y = y.unsqueeze(-1) * y_mask.unsqueeze(-1)
        t = t.unsqueeze(-1) * y_mask.unsqueeze(-1)
        t_sample = t_sample.unsqueeze(-1)

   
        branch_embedding_y = self.branch_embedding_y(y)
        branch_embedding_t = self.embedding_t_branch(t)
        trunk_embed = self.embedding_t_branch(t_sample)

       
        y_mask_enc = torch.where(y_mask == 1, False, True)


        branch_encoder_input = (torch.cat((branch_embedding_y , branch_embedding_t),dim=-1))

        #
        branch_encoder_output = self.branch_encoder(branch_encoder_input, src_key_padding_mask=y_mask_enc)
       # branch_encoder_output = self.branch_mlp(branch_encoder_output)


      #  branch_encoder_output = branch_encoder_output * y_mask.unsqueeze(-1)

        q = self.learnable_q.unsqueeze(0).expand(y.shape[0],-1,-1)


        branch_output,_ = self.branch_attention(q,branch_encoder_output,branch_encoder_output,key_padding_mask=y_mask_enc)
        branch_output = branch_output

  
        trunk_output = self.trunk_mlp(trunk_embed) 

        


        combined = torch.cat((branch_output.expand(-1,trunk_output.shape[1],-1),trunk_output),dim=-1)

        return self.final_proj(combined).squeeze()
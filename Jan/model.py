

import torch
import torch.nn as nn

class FIML(torch.nn.Module):

    def __init__(self, d_model,proj_dim=1024, heads=1):
        super(FIML, self).__init__()
        self.branch_embedding_y = nn.Sequential(nn.Linear(1, d_model))
      
        self.embedding_t_branch =   nn.Sequential(nn.Linear(1, d_model))
        self.embedding_t_trunk =   nn.Sequential(nn.Linear(1, d_model))

        self.branch_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=2*d_model, nhead=heads, batch_first=True), num_layers=6, enable_nested_tensor=False)
     
        
        self.branch_attention = nn.MultiheadAttention(d_model, 1 ,batch_first=True)

        self.learnable_q = nn.Parameter(torch.randn(1,d_model))

        self.trunk_mlp =   nn.Sequential(nn.Linear(d_model, d_model),nn.LeakyReLU(),
                                         nn.LayerNorm(d_model),
                                         nn.Linear(d_model, d_model),nn.LeakyReLU(),
                                         nn.Linear(d_model, d_model),nn.LeakyReLU(),
                                         nn.LayerNorm(d_model),
                                         nn.Linear(d_model, d_model),nn.LeakyReLU())  
        
        self.branch_mlp =   nn.Sequential(nn.Linear(2*d_model, 2*d_model),nn.LeakyReLU(),
                                         nn.LayerNorm(2*d_model),
                                         nn.Linear(2*d_model, d_model),nn.LeakyReLU(),
                                         nn.LayerNorm(d_model))
        
        self.final_proj = nn.Sequential(nn.Linear(2*d_model, proj_dim),nn.LeakyReLU(),
                                        nn.LayerNorm(proj_dim),
                                        nn.Linear(proj_dim, proj_dim),nn.LeakyReLU(),
                                        nn.Linear(proj_dim, proj_dim),nn.LeakyReLU(),
                                        nn.Linear(proj_dim, proj_dim),nn.LeakyReLU(negative_slope=0.1),
                                        nn.LayerNorm(proj_dim),
                                        nn.Linear(proj_dim, d_model),nn.LeakyReLU(negative_slope=0.1),
                                        nn.Linear(d_model,1),)
                                        
        
    def forward(self, y, t, t_sample, y_mask):
        y = y.unsqueeze(-1) * y_mask.unsqueeze(-1)
        t = t.unsqueeze(-1) * y_mask.unsqueeze(-1)
        t_sample = t_sample.unsqueeze(-1)

   
        branch_embedding_y = self.branch_embedding_y(y)
        branch_embedding_t = self.embedding_t_branch(t)
        trunk_embed = self.embedding_t_trunk(t_sample)

        y_mask_enc = torch.where(y_mask == 1, False, True)
      

        branch_encoder_input = (torch.cat((branch_embedding_y , branch_embedding_t),dim=-1))
   
        branch_encoder_output = self.branch_encoder(branch_encoder_input, src_key_padding_mask=y_mask_enc)
        branch_encoder_output = self.branch_mlp(branch_encoder_output)
 
        q = self.learnable_q.unsqueeze(0).expand(y.shape[0],-1,-1)
        branch_output,_ = self.branch_attention(q,branch_encoder_output,branch_encoder_output,key_padding_mask=y_mask_enc)

        trunk_output = self.trunk_mlp(trunk_embed) 
     


        combined = torch.cat((branch_output.expand(-1,trunk_output.shape[1],-1),trunk_output),dim=-1)

        return self.final_proj(combined).squeeze(), branch_output




class MegaTron(nn.Module):
    def __init__(self, d_model,fim_l_d_model=256, heads=1):
      super(MegaTron, self).__init__()
      self.local_scale_emb = nn.Linear(9,d_model)
      
      
      self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=d_model, nhead=heads, batch_first=True), num_layers=8, enable_nested_tensor=False)
      
      self.learnable_q = nn.Parameter(torch.empty(1,d_model))
      nn.init.xavier_uniform_(self.learnable_q)

      self.summary = nn.MultiheadAttention(d_model, heads ,batch_first=True)

      self.fim_l = FIML(d_model=fim_l_d_model,heads=4).to('cuda')
      self.fim_l.load_state_dict((torch.load("model_fim_l.pth")))
      self.fim_l = torch.compile(self.fim_l)

      for param in self.fim_l._orig_mod.parameters():
        param.requires_grad = False
      
      self.trunk_embed =  self.fim_l._orig_mod.embedding_t_trunk
      self.trunk_net =  self.fim_l._orig_mod.trunk_mlp


      self.extractor = nn.Sequential(nn.Linear(2*d_model, 4*d_model),nn.LeakyReLU(), 
                                      nn.LayerNorm(4*d_model),
                                      nn.Linear(4*d_model,2* d_model),nn.LeakyReLU(),
                                      nn.Linear(2*d_model, d_model),)
                                     
                                        
                                     
      self.final_proj = self.fim_l._orig_mod.final_proj

      self.sim_layer = nn.Sequential(nn.Linear(d_model , 4*d_model),nn.LeakyReLU(), 
                                      nn.LayerNorm(4*d_model),
                                      nn.Linear(4*d_model,4* d_model),nn.LeakyReLU(negative_slope=0.1),
                                      nn.Linear(4*d_model, d_model),nn.LeakyReLU(negative_slope=0.1),
                                      nn.Linear(d_model,d_model))

    @torch.no_grad()
    def getLastWindow(self,y, t, t_sample, y_mask,stats):
      stats_emb = self.local_scale_emb(stats).unsqueeze(1)
  
      
      _,out = self.fim_l(y, t, t_sample, y_mask) # 
      h_5 = torch.cat((out,stats_emb),dim=-1)

      return self.extractor(h_5)


       
    def forward(self,y, t, t_sample, y_mask,stats):

    #  h_5 = self.getLastWindow(y[:,4], t[:,4], t_sample[:,4], y_mask[:,4],stats[:,4])

      y, t, t_sample, y_mask,stats = y[:,:4], t[:,:4], t_sample[:,:5], y_mask[:,:4],stats[:,:4]

      stats_emb = self.local_scale_emb(stats)
  
      local_emb = None
      for i in range(0,4):
        y_i,t_i,t_sample_i,y_mask_i = y[:,i],t[:,i],t_sample[:,i],y_mask[:,i]
      
        _,out = self.fim_l(y_i, t_i, t_sample_i, y_mask_i) # 
        local_emb = out if local_emb == None else torch.cat((local_emb,out),dim=1)

      local_concat_emb = torch.cat((local_emb,stats_emb),dim=-1)
      local_concat_emb = self.extractor(local_concat_emb)



     

      megatron = self.transformer(local_concat_emb)

      learnable_q = self.learnable_q.unsqueeze(0).expand(y.shape[0],-1,-1)

      u,_ = self.summary(learnable_q,megatron,megatron)

     # u = self.sim_layer(u)

      cosine_sim =1# nn.functional.cosine_similarity(u.squeeze(1),h_5,dim=-1)

      last_window_t_embedd =self.trunk_embed( t_sample[:,-1].unsqueeze(-1))
      last_window_t_out =self.trunk_net(last_window_t_embedd)

      combined = torch.cat((u.expand(-1,last_window_t_out.shape[1],-1),last_window_t_out),dim=-1)

      out = self.final_proj(combined)


      return (out).squeeze(),cosine_sim
      
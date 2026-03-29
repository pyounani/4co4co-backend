import torch 
import torch.nn as nn 

class Emotic(nn.Module):
  ''' Emotic Model'''

  def __init__(self, model_path, context_dim=2048, body_dim=2048):
    self.target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.idle_device = torch.device("cpu")

    self.model = Emotic(num_context_features=context_dim, num_body_features=body_dim)
    checkpoint = torch.load(model_path, map_location="cpu")
    self.model.load_state_dict(checkpoint)
    self.model.to(self.idle_device)
    self.model.eval()

    
  def forward(self, x_context, x_body):
    context_features = x_context.view(-1, self.num_context_features)
    body_features = x_body.view(-1, self.num_body_features)
    fuse_features = torch.cat((context_features, body_features), 1)
    fuse_out = self.fc1(fuse_features)
    fuse_out = self.bn1(fuse_out)
    fuse_out = self.relu(fuse_out)
    fuse_out = self.d1(fuse_out)    
    cat_out = self.fc_cat(fuse_out)
    cont_out = self.fc_cont(fuse_out)
    return cat_out, cont_out

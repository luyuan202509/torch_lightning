import torch 
from torch import nn

class SimpleDenseNet(nn.Module):
    def __init__(self,input_size = 784,hidden_size=256,output_size=20) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size,hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size,hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size,hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size,output_size)
        )
    
    def forward(self,x:torch.Tensor)->torch.Tensor:
        batch_size,channels,width,height = x.size()
        x = x.view(batch_size,-1)
        return self.model(x)
        

import torch 
from torch.optim import optimizer
from  torchvision.datasets import MNIST
from torchvision.transforms import transforms
from torch.utils.data import DataLoader,Dataset

from pytorch_lightning import LightningModule,LightningDataModule,Trainer 
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.types import EVAL_DATALOADERS,TRAIN_DATALOADERS

from simple_dense_net import SimpleDenseNet

class MNISTModule(LightningModule):
    def __init__(self):
        super().__init__()
        self.data_dir = "../data"
        self.batch_size = 64
        self.model = SimpleDenseNet()
        self.criterion = torch.nn.CrossEntropyLoss()
        self.dataset = MNIST(self.data_dir, train=True, download=True, transform=transforms.ToTensor())

    def forward(self,x:torch.Tensor)->torch.Tensor:
        return self.model(x)

    # 单批次训练逻辑
    def training_step(self,batch:torch.Tensor,batch_idx:int)->torch.Tensor:
        features,lable = batch
        logit = self.forward(features)
        loss = self.criterion(logit,lable) # 单批次训练必须返回loss

        if batch_idx%10 == 0:
            self.log("train_loss",loss,prog_bar=True)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.SGD(self.model.parameters(),lr=0.01)
        return optimizer

    def train_dataloader(self) -> DataLoader:
        train_loader = DataLoader(self.dataset,batch_size=self.batch_size,shuffle=True)
        return train_loader

    
if __name__ == "__main__":
    model = MNISTModule()
    trainer = Trainer(max_epochs=10,accelerator="mps",devices=1)
    trainer.fit(model)

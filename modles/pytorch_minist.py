from numpy.random import f
import torch 
import numpy as np

from torch.optim import optimizer
from torch.utils.data import ConcatDataset,DataLoader,Dataset
#from torch.utils.data import ConcatDataset,DataLoader,Dataset,random_split
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
from simple_dense_net import SimpleDenseNet
import argparse


def main():
    model = SimpleDenseNet()
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--batch_size",type=int,default=64)
    argparser.add_argument("--lr",type=float,default=0.01)
    argparser.add_argument("--epochs",type=int,default=10)
    argparser.add_argument("--data_dir",type=str,default="../data")
    argparser.add_argument("--device",type=str,default="auto")
    args = argparser.parse_args()


    data_dir = args.data_dir
    lr = args.lr
    model = SimpleDenseNet()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(device)

    model = model.to(device)
    criterion = torch.nn.CrossEntropyLoss() # 损失函数
    optimizer = torch.optim.SGD(model.parameters(),lr=lr)
    
    train_dataset = MNIST(data_dir,train=True,transform=transforms.ToTensor())
    train_loader = DataLoader(train_dataset,batch_size=args.batch_size,shuffle=True)
    # test_dataset = MNIST(data_dir,train=False,transform=transforms.ToTensor())
    # test_loader = Dtaloader(test_dataset,batch_size=args.batch_size,shuffle=True)

    for epoch in range(args.epochs):
        for data in train_loader:
            inputs,labels = data
            inputs,labels = inputs.to(device),labels.to(device)
            optimizer.zero_grad() # 清空梯度
            outputs = model(inputs)
            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()
        print(f"epoch:{epoch},loss:{loss.item()}")



if __name__ == "__main__":
    main()
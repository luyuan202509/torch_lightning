"""
1、数据部分从lightning Moudle 单独分离
2、模型逻辑加上validate,test 部分
3、除记录损失值外，记录准确率

"""
import torch
from torchmetrics import Accuracy, MeanMetric, MaxMetric
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split

from pytorch_lightning import LightningModule, LightningDataModule, Trainer
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS

from simple_dense_net import SimpleDenseNet


class MNISTModule(LightningModule):
    def __init__(self):
        super().__init__()
        self.model = SimpleDenseNet()
        self.criterion = torch.nn.CrossEntropyLoss()

        # metrix
        self.train_acc = Accuracy(task="multiclass", num_classes=10)
        self.val_acc = Accuracy(task="multiclass", num_classes=10)
        self.test_acc = Accuracy(task="multiclass", num_classes=10)

        # loss
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # 最好 loss
        self.val_acc_best = MaxMetric()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    # ============================================================
    # =================训练过程相关==================================
    # ============================================================
    # 训练过程钩子函数，每个batch开始，或结束，每个epoch开始或结束
    def on_train_start(self) -> None:
        self.val_loss.reset()
        self.val_acc.reset()
        self.val_acc_best.reset()

    def model_step(self, batch):
        features, label = batch
        logits = self.forward(features)
        loss = self.criterion(logits, label)
        predict = torch.argmax(logits, dim=1)

        return loss, predict, label

    # 单批次训练逻辑
    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        loss, predict, label = self.model_step(batch)
        self.train_loss(loss)
        self.train_acc(predict, label)

        self.log("train/loss", self.train_loss, on_step=True, on_epoch=False, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=True, on_epoch=False, prog_bar=True)
        return loss

    def on_train_end(self) -> None:
        pass

    def on_train_epoch_end(self) -> None:
        pass

    # ============================================================
    # =================验证过程相关=================================
    # ============================================================
    # 单批次验证逻辑
    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        loss, predict, label = self.model_step(batch)
        self.val_loss(loss)
        self.val_acc(predict, label)

        self.log("val/loss", self.val_loss, on_step=True, on_epoch=False, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=True, on_epoch=False, prog_bar=True)
        return loss

    def on_validation_epoch_end(self) -> None:
        acc = self.val_acc.compute()
        self.val_acc_best(acc)
        self.log("val/acc_best", acc, sync_dist=True, prog_bar=True)

    # ============================================================
    # =================测试过程相关=================================
    # ============================================================
    # 单批次测试逻辑
    def test_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        loss, preds, label = self.model_step(batch)
        self.test_loss(loss)
        self.test_acc(preds, label)

        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        return optimizer

    # ============================================================
    # =================其他相关=================================
    # ============================================================
    # def train_dataloader(self) -> DataLoader:
    #     train_loader = DataLoader(self.dataset,batch_size=self.batch_size,shuffle=True)
    #     return train_loader


class MNISTDataModule(LightningDataModule):
    def __init__(self, data_dir: int, batch_size: int):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

    def prepare_data(self) -> None:
        # 数据下载等单次执行的任务
        pass

    def setup(self, stage: str) -> None:
        dataset = MNIST(self.data_dir, download=False, transform=transforms.ToTensor())
        self.data_train, self.data_val, self.data_test = random_split(
            dataset=dataset,
            lengths=[0.8, 0.1, 0.1],
            generator=torch.Generator().manual_seed(1024))

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        train_dataloader = DataLoader(self.data_train, batch_size=self.batch_size, shuffle=True)
        return train_dataloader

    def val_dataloader(self) -> EVAL_DATALOADERS:
        val_dataloader = DataLoader(self.data_val, batch_size=self.batch_size, shuffle=True)
        return val_dataloader

    def test_dataloader(self) -> EVAL_DATALOADERS:
        test_dataloader = DataLoader(self.data_test, batch_size=self.batch_size, shuffle=True)
        return test_dataloader

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        pass


if __name__ == "__main__":
    data_dir = "../data"
    model = MNISTModule()
    datamodule = MNISTDataModule("../data", 64)
    #trainer = Trainer(max_epochs=5, accelerator="mps", devices=1)
    #trainer = Trainer(max_epochs=2, accelerator="mps", devices=1, enable_model_summary=False) # enable_model_summary 控制训练前对模型参数量，模型结构等总结日志打印信息
    #trainer = Trainer(max_epochs=2, accelerator="mps", devices=1,enable_progress_bar=True) # enable_progress_bar 训练进度条显示与否控制
    #trainer = Trainer(max_epochs=2, accelerator="mps", devices=1,num_sanity_val_steps=2) # num_sanity_val_steps 冒烟测试，先从从valid dataloader中取两个batch，前向传播测试
    #trainer = Trainer(max_epochs=2, accelerator="mps", devices=1,enable_checkpointing=False)  # enable_checkpointing 控制每次训练完成保存checkpoint
    # trainer = Trainer(max_epochs=2, accelerator="mps", devices=1,logger=False)  # logger 控制每次训练完是否保存 非控制台 文件格式日志内容
    #trainer = Trainer(max_epochs=2, accelerator="mps", devices=1,default_root_dir="../lightning_data")  # default_root_dir 控制训练checkpoint,日志等等保存路径

    #trainer = Trainer(max_epochs=2, accelerator="mps", devices=1) # accelerator="mps", devices=1, 控制多卡训练，详见官网文档
    #trainer = Trainer(max_epochs=2, accelerator="mps", devices=1,precision='32-true') # precision 控制训练权重参数精度
    #trainer = Trainer(max_steps=2, accelerator="mps", devices=1) # max_steps 训练最大的步数，所有epoch步数 大于 这个最大步数时候会提前结束训练，优先级比epoch更高
    #trainer = Trainer(max_epochs=5, accelerator="mps", devices=1,check_val_every_n_epoch=2) # check_val_every_n_epoch 控制多少epoch训练完成进行验证
    trainer = Trainer(max_epochs=5, accelerator="mps", devices=1,val_check_interval=500) # val_check_interval 控制多少 batch 训练完成进行验证

    print("开始训练。。。")
    trainer.fit(model=model, datamodule=datamodule)
    # print("开始验证。。。")
    # trainer.validate(model=model,datamodule = datamodule)
    # print("开始测试。。。")
    # trainer.test(model=model, datamodule=datamodule)

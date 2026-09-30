"""
-- tensorbord 日志记录 版本
"""
import torch
from lightning.pytorch.loggers import TensorBoardLogger, tensorboard
from torchmetrics import Accuracy, MeanMetric, MaxMetric
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split

from pytorch_lightning import LightningModule, LightningDataModule, Trainer
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS

from simple_dense_net import SimpleDenseNet


class MNISTModule(LightningModule):
    def __init__(self,name = 'Tom',age = 11):
        super().__init__()
        self.model = SimpleDenseNet()
        self.criterion = torch.nn.CrossEntropyLoss()
        self.save_hyperparameters() # 保存模型超参,保存到hparams.yml文件

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

        self.log_dict(
             dictionary={
                "train/loss":self.train_loss,
                "train/acc":self.train_acc,
                 # 获取其他评测指标
            }, on_step=True, on_epoch=False, prog_bar=True
        )

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
    model = MNISTModule()
    datamodule = MNISTDataModule("../data", 64)

    #tensorboard 记录训练日志，命令行查看：tensorboard --logdir /Users/luyuan/neulife/pyproject/torch_lightning/modles/tensorBordLogs
    logger = TensorBoardLogger("tensorBordLogs",name="mnist")

    trainer = Trainer(max_epochs=2, accelerator="mps", devices=1,logger=logger)

    # print("开始训练。。。")
    # trainer.fit(model=model, datamodule=datamodule)

    # print("开始验证。。。")
    # trainer.validate(model=model,datamodule = datamodule)

    checkpoint_path = "/Users/luyuan/neulife/pyproject/torch_lightning/modles/lightning_logs/version_0/checkpoints/epoch=4-step=3750.ckpt"
    model = MNISTModule.load_from_checkpoint(checkpoint_path)
    print("开始测试。。。")
    trainer.test(model=model, datamodule=datamodule)

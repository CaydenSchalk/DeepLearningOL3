import numpy as np
import torch
import torch.nn as nn
import lightning.pytorch as pl
import torchmetrics
import torchvision
from torchinfo import summary
from torchview import draw_graph
from IPython.display import display
import sympy as sp
sp.init_printing(use_latex=True)
import contextlib
import pandas as pd
import matplotlib.pyplot as plt
import logging
import wandb
import json
import warnings
from pathlib import Path
import argparse
import os

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
TYPE = "gelu" # gelu prerelu pregelu
EXPERIMENT = Path(__file__).name
PROJECT = "OL3-CIFAR100"
DATASET = "CIFAR100"
CUSTOM = False
SILENT = False

output_context = (
    contextlib.redirect_stdout(open(os.devnull, "w"))
    if SILENT
    else contextlib.nullcontext()
)

error_context = (
    contextlib.redirect_stderr(open(os.devnull, "w"))
    if SILENT
    else contextlib.nullcontext()
)

def convert_relu_to_gelu(model):
    for child_name, child in model.named_children():
        if isinstance(child, torch.nn.ReLU):
            setattr(model, child_name, torch.nn.GELU())
        else:
            convert_relu_to_gelu(child)

class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(self,
                 batch_size = 128,
                 val_split = 0.2,
                 num_workers = 8,
                 location = "./datasets",
                 **kwargs):
        super().__init__(**kwargs)
        self.batch_size = batch_size
        self.val_split = val_split
        self.num_workers = num_workers
        self.location = location
        self.input_shape = None
        self.output_shape = None
        self.data_train = None
        self.data_val = None
        self.data_test = None

    def setup(self, stage: str):
        if (stage == 'fit' or \
            stage == 'validate') and \
                not(self.data_train and self.data_val):
            if DATASET == "CIFAR10":
                training_dataset = torchvision.datasets.CIFAR10(root='./datasets',download=True, train=True)
            else:
                training_dataset = torchvision.datasets.CIFAR100(root='./datasets',download=True,train=True)

            x_train = training_dataset.data.transpose((0,3,1,2))
            y_train = np.array(training_dataset.targets)
            self.input_shape = x_train.shape[1:]
            self.output_shape = (len(np.unique(y_train)),)
            rng = np.random.default_rng()
            permutation = rng.permutation(x_train.shape[0])
            split_point = int(x_train.shape[0]*(1.0-self.val_split))
            self.data_train = list(zip(torch.Tensor(x_train[permutation[:split_point]]).to(torch.float32),
                                       torch.Tensor(y_train[permutation[:split_point]]).to(torch.long)))
            self.data_val = list(zip(torch.Tensor(x_train[permutation[split_point:]]).to(torch.float32),
                                     torch.Tensor(y_train[permutation[split_point:]]).to(torch.long)))
        if (stage == 'test' or \
            stage == 'predict') and \
                not self.data_test:
            if DATASET == "CIFAR10":
                testing_dataset = torchvision.datasets.CIFAR10(root='./datasets',download=True, train=False)
            else:
                testing_dataset = torchvision.datasets.CIFAR100(root='./datasets',download=True,train=False)

            x_test = testing_dataset.data.transpose((0,3,1,2))
            y_test = np.array(testing_dataset.targets)
            self.input_shape = x_test.shape[1:]
            self.output_shape = (len(np.unique(y_test)),)
            self.data_test = list(zip(torch.Tensor(x_test).to(torch.float32),
                                      torch.Tensor(y_test).to(torch.long)))

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.data_train,
                                           batch_size=self.batch_size,
                                           num_workers=self.num_workers,
                                           pin_memory=True,
                                           persistent_workers=True,
                                           shuffle=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.data_val,
                                           batch_size=self.batch_size,
                                           num_workers=self.num_workers,
                                           pin_memory=True,
                                           persistent_workers=True,
                                           shuffle=False)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.data_test,
                                           batch_size=self.batch_size,
                                           num_workers=self.num_workers,
                                           pin_memory=True,
                                           persistent_workers=True,
                                           shuffle=False)

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(self.data_test,
                                           batch_size=self.batch_size,
                                           num_workers=self.num_workers,
                                           pin_memory=True,
                                           persistent_workers=True,
                                           shuffle=False)

# Define Trainable Module (Abstract Base Class)
class LightningNetwork(pl.LightningModule):
    def __init__(self, **kwargs):
        # This is the contructor, where we typically make
        # layer objects using provided arguments.
        super().__init__(**kwargs) # Call the super class constructor

    def predict_step(self, predict_batch, batch_idx):
        x, y_true = predict_batch
        y_pred = self.predict(x)
        return y_pred, y_true

    def training_step(self, train_batch, batch_idx):
        x, y_true = train_batch
        y_pred = self(x)
        for metric_name, metric_function in self.network_metrics.items():
            metric_value = metric_function(y_pred,y_true)
            self.log('train_'+metric_name, metric_value, on_step=False, on_epoch=True)
        loss = self.network_loss(y_pred,y_true)
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y_true = val_batch
        y_pred = self(x)
        for metric_name, metric_function in self.network_metrics.items():
            metric_value = metric_function(y_pred,y_true)
            self.log('val_'+metric_name, metric_value, on_step=False, on_epoch=True)
        loss = self.network_loss(y_pred,y_true)
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        return loss

    def testing_step(self, test_batch, batch_idx):
        x, y_true = test_batch
        y_pred = self(x)
        for metric_name, metric_function in self.network_metrics.items():
            metric_value = metric_function(y_pred,y_true)
            self.log('test_'+metric_name, metric_value, on_step=False, on_epoch=True)
        loss = self.network_loss(y_pred,y_true)
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        return loss

# Attach loss, metrics, and optimizer
class MultiClassNetwork(LightningNetwork):
    def __init__(self,
                 num_classes,
                 **kwargs):
        # This is the contructor, where we typically make
        # layer objects using provided arguments.
        super().__init__(**kwargs) # Call the super class constructor

        # This creates an accuracy function
        self.network_metrics = torch.nn.ModuleDict({
            'acc': torchmetrics.classification.Accuracy(task='multiclass',
                                                        num_classes=num_classes)
        })
        # This creates a loss function
        self.network_loss = torch.nn.CrossEntropyLoss()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001, fused=True)
        return optimizer

# Attach standardization and augmentation
class StandardizeTransform(MultiClassNetwork):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        # Source: https://pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html
        # Needs to always be applied to any incoming
        # image for this model. The Compose operation
        # takes a list of torchvision transforms and
        # applies them in sequential order, similar
        # to neural layers...
        self.standardize = torchvision.transforms.Compose([
            torchvision.transforms.Resize([232]),
            torchvision.transforms.CenterCrop([224]),
            torchvision.transforms.Lambda(lambda x: x / 255.0),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225]),
        ])
        # Besides just standardization, the images can also undergo
        # augmentation using torchvision. Again, we compose
        # these operations together - ranges are provided for
        # each of these augmentations.
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomAffine(degrees=(-10.0,10.0),
                                                translate=(0.1,0.1),
                                                scale=(0.9,1.1),
                                                shear=(-10.0,10.0)),
            torchvision.transforms.RandomHorizontalFlip(0.5),
        ])

    def forward(self, x):
        y = x
        y = self.standardize(y)
        if self.training:
            y = self.transform(y)
        return y

# Our actual network which inherits all of the
# above functionality - note we can utilize
# super().forward() to functionalize the parent class
# calculations directly.
class ResNet50(StandardizeTransform):
    def __init__(self,
                 input_shape,
                 output_shape, network_type,
                 **kwargs):
        super().__init__(num_classes=output_shape[-1],
                         **kwargs)
        self.save_hyperparameters()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.output_activation = torch.nn.Softmax(dim=-1)
        self.network_type = network_type

        # Replace output layer (default it 1000) with 10 units
        if self.network_type == 'prerelu':
            self.resnet = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
        elif self.network_type == 'pregelu':
            self.resnet = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
            convert_relu_to_gelu(self.resnet)
        elif self.network_type == 'relu':
            self.resnet = torchvision.models.resnet50(weights=None)
        elif self.network_type == 'gelu':
            self.resnet = torchvision.models.resnet50(weights=None)
            convert_relu_to_gelu(self.resnet)

        # self.resnet = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
        self.resnet.fc = torch.nn.Linear(self.resnet.fc.in_features,
                                         output_shape[-1])

    def forward(self, x):
        y = x
        y = super().forward(y)
        y = self.resnet(y)
        return y

    def predict(self, x):
        return self.output_activation(self(x))

def main():
    parser = argparse.ArgumentParser()

    os.environ["WANDB_SILENT"] = "true"
    os.environ["WANDB_CONSOLE"] = "off"
    all_acc, all_loss = [], []

    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--data-dir', type=Path, default=HERE / 'datasets')
    parser.add_argument('--output-dir', type=Path, default=HERE)

    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = args.output_dir / 'runs' / EXPERIMENT
    run_dir.mkdir(parents=True, exist_ok=True)
    data_module = CIFAR10DataModule()
    data_module.setup('validate')

    for run in range(args.runs):

        with output_context, error_context:
            pl.seed_everything(args.seed + run, workers=True, verbose=False)

            model = ResNet50(input_shape=data_module.input_shape,
                             output_shape=data_module.output_shape, network_type=TYPE)

            summary(model)

            logger = pl.loggers.CSVLogger(str(run_dir / "logs"), name=f'run-{run + 1:02d}')
            wandb_logger = pl.loggers.WandbLogger(
                project=PROJECT, name=f'{EXPERIMENT}-run{run + 1}',
                group=EXPERIMENT, save_dir=str(run_dir)
            )

            loggers = []
            loggers.append(wandb_logger)
            loggers.append(logger)
            configuration = {
                'experiment': EXPERIMENT, 'runs': args.runs, 'epochs': args.epochs,
                'seed': args.seed, 'split_seed': 0, 'batch_size': 128,
                'custom': CUSTOM, 'data_dir': str(args.data_dir.resolve()),
            }

            configuration = json.loads(json.dumps(configuration))
            wandb_logger.log_hyperparams({**configuration, 'run': run + 1,
                                          'run_seed': args.seed + run})
            trainer = pl.Trainer(max_epochs=50,logger=loggers,
                                 accelerator='gpu',
                                 precision="16-mixed",
                                 benchmark=True,
                                 enable_progress_bar=True,
                                 enable_checkpointing=False,
                                 log_every_n_steps=0,
                                 callbacks=[pl.callbacks.RichProgressBar(refresh_rate=20)])



            # result = trainer.validate(model, dataloaders=data_module)

            trainer.fit(model, data_module)
            # result = trainer.validate(model, data_module)
            wandb.finish()

        results = pd.read_csv(logger.log_dir+"/metrics.csv",delimiter=',')

        print("Validation accuracy:",*["%.8f"%(x) for x in
                                       results['val_acc'][np.logical_not(np.isnan(results["val_acc"]))]])
        print("Validation loss:",*["%.8f"%(x) for x in
                                   results['val_loss'][np.logical_not(np.isnan(results["val_loss"]))]])

        accuracy = results['val_acc'][np.logical_not(np.isnan(results['val_acc']))].to_numpy()
        loss = results['val_loss'][np.logical_not(np.isnan(results['val_loss']))].to_numpy()
        all_acc.append(accuracy)
        all_loss.append(loss)

    # grab everything after it's finished
    for metric, curves in [('acc', all_acc), ('loss', all_loss)]:
        target = args.output_dir / f'{EXPERIMENT}-{metric}-results.txt'
        np.savetxt(target, np.array(curves), fmt='%.8f')

if __name__ == '__main__':
    main()
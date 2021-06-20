import os
import ast
import torch
import torch.nn as nn
import torch.optim as optim

from argparse import ArgumentParser
from torch.nn import functional as F
from torch.utils.data import DataLoader

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.core.lightning import LightningModule

from model import SpeechRecognition
from dataset import Data, collate_fn_padd
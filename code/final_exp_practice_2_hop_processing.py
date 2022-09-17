"""Experiment for Practice Processing"""

import os
import random

import numpy as np
import torch
from gensim.models import FastText
from torch import nn, optim
from torch.utils.data import DataLoader

from code_flow.dataset import ASTSeqDataset
from code_flow.experiment import BinaryclassExperiment
from code_flow.model_zoo import AST2Class
from config import processing_2_hop_settings as practice_settings

# Seeds
seeds = practice_settings.seeds
os.environ['PYTHONHASHSEED'] = seeds['PYTHONHASHSEED']
torch.manual_seed(seeds['TORCH_MANNUAL_SEED'])
random.seed(seeds['PY_RANDOM_SEED'])
np.random.seed(seeds['NP_RANDOM_SEED'])

# Filepaths
filepaths = practice_settings.filepaths
dataset = filepaths.dataset
model_path = filepaths.model_path

# Labels
labels = practice_settings.labels

# Dataset Config
dataset_config = practice_settings.dataset
num_paths = dataset_config.num_paths
split_ratio = list(dataset_config.split_ratio)
bs = dataset_config.bs

dataset = ASTSeqDataset(filename=dataset,
                        split_ratio=split_ratio,
                        length=num_paths,
                        labels=labels,
                        randomize=True)

print("Datasets created")
print("Creating dataloader")

training_dataset = dataset.get_set('train')
val_dataset = dataset.get_set('valid')
test_dataset = dataset.get_set('test')


train_loader = DataLoader(training_dataset,
                          batch_size=bs,
                          drop_last=True)

valid_loader = DataLoader(val_dataset,
                          batch_size=bs,
                          drop_last=True)

test_loader = DataLoader(test_dataset,
                         batch_size=bs,
                         drop_last=True)

embedding = FastText.load(model_path)

# Model Config

model_config = practice_settings.model

n_vocab = len(dataset.vocab)
n_embed = model_config.n_embed
n_hidden = model_config.n_hidden
n_output = model_config.n_output
n_layers = model_config.n_layers

model = AST2Class(n_vocab,
                   n_embed,
                   n_hidden,
                   n_output,
                   n_layers,
                   batch_size=bs)

print("Model created")
print("Loading embedding")

model.load_embedding(dataset.vocab, embedding.wv)


# Experiment Config
training_config = practice_settings.training

epochs = training_config.epochs
print_every = training_config.print_every
max_clip = training_config.max_clip
learning_rate = training_config.learning_rate
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

print("Running Experiment")

experiment = BinaryclassExperiment(
    name='Practice_Processing_2_Hop',
    model=model,
    train_dataloader=train_loader,
    valid_dataloader=valid_loader,
    test_dataloader=test_loader,
    optimizer=optimizer,
    criterion=criterion,
    epochs=epochs,
    max_clip=max_clip,
    config=practice_settings,
    dirpath='experiment_results/')

import matplotlib 
import matplotlib.pyplot as plt
from utils import replot_conf_matrix

if os.environ['EXPERIMENT'] == '1':
    experiment.run(print_every)

if os.environ['EVAL'] == '1':
    replot_conf_matrix(experiment)
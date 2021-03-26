"""
This example loads the pre-trained SentenceTransformer model 'bert-base-nli-mean-tokens' from the server.
It then fine-tunes this model for some epochs on the STS benchmark dataset.
Note: In this example, you must specify a SentenceTransformer model.
If you want to fine-tune a huggingface/transformers model like bert-base-uncased, see training_nli.py and training_stsbenchmark.py
"""
from torch.utils.data import DataLoader
import math
from sentence_transformers import SentenceTransformer, LoggingHandler, losses, util, InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
import logging
from datetime import datetime
import os
import gzip
import csv
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle
import torch
from utils import Config

config = Config(config_file='/data/yuchen/projects/sentence_embedding/config/train.json')

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])


# Read the dataset
model_name = 'distilbert-base-nli-mean-tokens'
train_batch_size = 64
num_epochs = 4
# model_save_path = 'output/new-pos-samples-pos2million-neg2million-level3'+model_name+'-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
model_save_path = 'output/test'

print(model_save_path)


# Load a pre-trained sentence transformer model
# device = torch.device("cuda:1")
model = SentenceTransformer(model_name,device="cuda:1")

# Convert the dataset to a DataLoader ready for training
logging.info("Read STSbenchmark train dataset")

train_samples = []
dev_samples = []
test_samples = []

train_samples = pickle.load(open(config['train_data'],'rb'))
dev_samples = pickle.load(open(config['valid_data'],'rb'))

train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)
train_loss = losses.CosineSimilarityLoss(model=model)


# Development set: Measure correlation between cosine score and gold labels
logging.info("Read STSbenchmark dev dataset")
evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, name='sts-dev')


# Configure the training. We skip evaluation in this example
warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1) #10% of train data for warm-up
logging.info("Warmup-steps: {}".format(warmup_steps))



# Train the model
model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=evaluator,
          epochs=num_epochs,
          evaluation_steps=5000,
          warmup_steps=warmup_steps,
          output_path=model_save_path)


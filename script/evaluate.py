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
##############################################################################
#
# Load the stored model and evaluate its performance on STS benchmark dataset
#
##############################################################################
model_save_path = '/data/yuchen/projects/sentence_embedding/output/training_stsbenchmark_continue_training-distilbert-base-nli-mean-tokens-2021-03-09_17-56-07'
dev_samples = pickle.load(open('data/train/final/valid_4millon.pkl','rb'))

test_samples = dev_samples[1000:2000]
model = SentenceTransformer(model_save_path)
test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, name='sts-test')
test_evaluator(model, output_path=model_save_path)
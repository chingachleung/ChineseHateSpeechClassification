# Importing the libraries needed
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import torch
from torch import cuda
import argparse
import logging

from HateSpeechData import HateSpeechData
from BertClass import BertClass
from model_train import *

logging.basicConfig(level=logging.ERROR)
args = argparse.ArgumentParser(description='hate speecch classification using BERT')
args.add_argument('-a', '--train_file', type=str, help='train file', required=True)
args.add_argument('-v', '--val_file', type=str, help='val file', required=True)
args = args.parse_args()

#global variables

MAX_LEN = 100
BATCH_SIZE = 15
EPOCHS = 10

if __name__=="__main__":

    # Setting up the device for GPU usage
    device = 'cuda' if cuda.is_available() else 'cpu'
    model = BertClass()
    #for gpu further training
    #model = nn.DataParallel(model, device_ids=[0,1,2,3])
    model.to(device)

    train_file = args.train_file
    val_file = args.val_file


    train_params = {'batch_size': BATCH_SIZE,
                    'shuffle': True,
                    'num_workers': 0
                    }

    #load the data
    train_data = pd.read_csv(train_file, encoding='utf-8-sig')
    val_data = pd.read_csv(val_file, encoding='utf-8-sig')

    tweets = train_data["Tweet"]
    categories = train_data["Label"]

    val_tweets = val_data["Tweet"]
    val_categories = val_data["Label"]

    tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese", truncation=True)

    #data preparation
    onehot_encoder = OneHotEncoder(handle_unknown="ignore")  # set to zeros if new categories in test set occur

    training_onehot_targets = onehot_encoder.fit_transform(categories.values.reshape(-1, 1)).toarray()
    training_set = HateSpeechData(tweets, training_onehot_targets, tokenizer, MAX_LEN)
    training_loader = DataLoader(training_set, **train_params)

    val_onehot_targets = onehot_encoder.transform(val_categories.values.reshape(-1, 1)).toarray()
    val_set = HateSpeechData(val_tweets, val_onehot_targets, tokenizer, MAX_LEN)
    val_loader = DataLoader(val_set)

    #start training
    model, loss, optimizer = train(model, EPOCHS, training_loader, val_loader,onehot_encoder)
    #save the model
    torch.save({'epoch': EPOCHS,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'encoder': onehot_encoder},'bert_model_test3.pt')

    print('All files saved')
import pandas as pd
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import torch
from torch import cuda
import argparse
import logging

from HateSpeechData import HateSpeechData
from BertClass import BertClass
from model_train2 import *

logging.basicConfig(level=logging.ERROR)
args = argparse.ArgumentParser(description='hate speecch classification using BERT')
args.add_argument('-a', '--train_file', type=str, help='train file', required=True)
args.add_argument('-v', '--val_file', type=str, help='val file', required=True)
args = args.parse_args()

#global variables

MAX_LEN = 100
BATCH_SIZE = 15
EPOCHS = 8
LEARNING_RATE = 1e-05

if __name__=="__main__":

    # Setting up the device for GPU usage
    device = 'cuda' if cuda.is_available() else 'cpu'
    model = BertClass()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
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
    #labels: 0:Neithr, 1: Abusive-only, 2: Hate-speech
    tweets = train_data["Tweet"]
    categories = train_data["Label"].astype(int)

    val_tweets = val_data["Tweet"]
    val_categories = val_data["Label"].astype(int)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese", truncation=True)

    training_set = HateSpeechData(tweets, categories, tokenizer, MAX_LEN)
    training_loader = DataLoader(training_set, **train_params)


    #apparently pytorch does one-hot encoding itself if you pass the class index
    val_set = HateSpeechData(val_tweets, val_categories, tokenizer, MAX_LEN)
    val_loader = DataLoader(val_set)

    #start training
    train(model, optimizer, EPOCHS, training_loader, val_loader)


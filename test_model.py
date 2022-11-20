from tqdm import tqdm
import torch
from torch import cuda
import argparse
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix,ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import DataLoader
from HateSpeechData import HateSpeechData
from BertClass import BertClass
import torch.nn as nn
from transformers import AutoTokenizer
from sklearn.preprocessing import OneHotEncoder


from validation import valid
device = 'cuda' if cuda.is_available() else 'cpu'

args = argparse.ArgumentParser(description='validating the Roberta model')
args.add_argument('-a', '--testing_file', type=str, help='testing_file', required=True)
args.add_argument('-m', '--model_file', type=str, help='saved model', required=True)
args = args.parse_args()

LEARNING_RATE = 1e-05
MAX_LEN = 100 # matching the training parameters


def make_confusion_matrix(labels, predictions):
    fig, ax2 = plt.subplots(figsize=(14, 12))

    label_names = sorted(set(labels))
    cm = confusion_matrix(labels, predictions, labels=label_names)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=label_names)
    disp.plot(ax=ax2)
    plt.show()
    plt.savefig('testing_cm.png')


if __name__=="__main__":

    model_file = args.model_file
    testing_file = args.testing_file

    model = BertClass()
    #model = nn.DataParallel(model, device_ids=[0,1,2,3])
    #model = nn.DataParallel(model, device_ids=[0])
    model.to(device)

    loss_function = torch.nn.CrossEntropyLoss()
    tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese", truncation=True)
    onehot_encoder = OneHotEncoder(handle_unknown="ignore")

    #optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
    checkpoint = torch.load(model_file)
    model.load_state_dict(checkpoint['model_state_dict'])
    #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #onehot_encoder = checkpoint['encoder']

    test_data = pd.read_csv(testing_file, encoding='utf-8-sig')
    testing_onehot_targets = onehot_encoder.transform(test_data['Label'].values.reshape(-1, 1)).toarray()
    testing_set = HateSpeechData(test_data['Tweet'], testing_onehot_targets, tokenizer, MAX_LEN)
    testing_loader = DataLoader(testing_set)

    epoch_loss, predictions, labels = valid(model, testing_loader,onehot_encoder)

    scores = classification_report(labels, predictions)
    print(scores)

    make_confusion_matrix(labels, predictions)



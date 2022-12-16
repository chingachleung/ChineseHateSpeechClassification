import math
import torch
from torch import cuda
import argparse
from sklearn.metrics import classification_report, confusion_matrix,ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import DataLoader
from HateSpeechData import HateSpeechData
from BertClass import BertClass
from transformers import AutoTokenizer
from validation2 import valid
device = 'cuda' if cuda.is_available() else 'cpu'
import numpy as np
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
    plt.rcParams.update({'font.size': 22})
    plt.show()
    plt.savefig('testing_cm.png')


if __name__=="__main__":

    model_file = args.model_file
    testing_file = args.testing_file
    label_dict = {0:"Neither", 1: "Abusive-only", 2: "Hate-speech"}
    class_weights = [math.sqrt(17/61), math.sqrt(17/17), math.sqrt(17/22)] # map 0 to 0.3, 1 t0 0.4...
    weight_dict = torch.tensor(class_weights).to(device)
    loss_function = torch.nn.CrossEntropyLoss(weight=weight_dict)

    model = BertClass()
    #model = nn.DataParallel(model, device_ids=[0,1,2,3])
    #model = nn.DataParallel(model, device_ids=[0])
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese", truncation=True)
    #onehot_encoder = OneHotEncoder(handle_unknown="ignore")
    #optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
    checkpoint = torch.load(model_file)
    model.load_state_dict(checkpoint['model_state_dict'])
    #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #onehot_encoder = checkpoint['encoder']

    test_data = pd.read_csv(testing_file, encoding='utf-8-sig')
    #testing_onehot_targets = onehot_encoder.fit_transform(test_data['Label'].values.reshape(-1, 1)).toarray()
    targets = test_data["Label"].astype(int)
    tweets = test_data['Tweet']
    testing_set = HateSpeechData(tweets, targets, tokenizer, MAX_LEN)
    testing_loader = DataLoader(testing_set)

    epoch_loss, predictions, labels = valid(model, testing_loader,loss_function)
    for p in predictions:
      print(p)
   
    labels = np.vectorize(label_dict.get)(labels)
    predictions = np.vectorize(label_dict.get)(predictions)
    scores = classification_report(labels, predictions)
    print(scores)

    make_confusion_matrix(labels, predictions)


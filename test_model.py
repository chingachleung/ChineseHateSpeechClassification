
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
import numpy as np
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
    plt.rcParams.update({'font.size': 16})
    plt.show()
    plt.savefig('testing_cm.png')


if __name__=="__main__":

    model_file = args.model_file
    testing_file = args.testing_file
    label_dict = {0:"Neither", 1: "Abusive-only", 2: "Hate-speech"}
    class_weights = [0.3, 0.4, 0.3] # map 0 to 0.3, 1 t0 0.4...
    weight_dict = torch.tensor(class_weights).to(device)
    loss_function = torch.nn.CrossEntropyLoss(weight=weight_dict)

    model = BertClass()
    #model = nn.DataParallel(model, device_ids=[0,1,2,3])
    #model = nn.DataParallel(model, device_ids=[0])
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese", truncation=True)

    checkpoint = torch.load(model_file)
    model.load_state_dict(checkpoint['model_state_dict'])
    test_data = pd.read_csv(testing_file, encoding='utf-8-sig')
    targets = test_data["Label"].astype(int)
    testing_set = HateSpeechData(test_data['Tweet'],targets, tokenizer, MAX_LEN)
    testing_loader = DataLoader(testing_set)

    epoch_loss, predictions, labels = valid(model, testing_loader,loss_function)
    for p in predictions:
      print(p)
    print("real labels: ########################################################")
    for l in labels:
      print(l)
    labels = np.vectorize(label_dict.get)(labels)
    predictions = np.vectorize(label_dict.get)(predictions)
    scores = classification_report(labels, predictions)
    print(scores)

    make_confusion_matrix(labels, predictions)



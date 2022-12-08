import torch
from tqdm import tqdm
from torch import cuda
from validation2 import valid
import numpy as np

device = 'cuda' if cuda.is_available() else 'cpu'
from sklearn.metrics import f1_score

LEARNING_RATE = 1e-05

class EarlyStopper:
    def __init__(self, patience=2, min_delta=-0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < (self.min_validation_loss + self.min_delta):
            self.min_validation_loss = validation_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False



def train(model,optimizer,epochs,training_loader, val_loader):
    model.to(device)
    class_weights = [0.3, 0.4, 0.3] # map 0 to 0.3, 1 t0 0.4...
    weight_dict = torch.tensor(class_weights).to(device)
    loss_function = torch.nn.CrossEntropyLoss(weight=weight_dict)
    early_stopper = EarlyStopper()
    model.train()

    for epoch in range(epochs):
        print(f"epoch {epoch}")
        tr_loss = 0
        n_correct = 0
        nb_tr_steps = 0
        nb_tr_examples = 0
        for _, data in tqdm(enumerate(training_loader, 0)):
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            targets = data['targets'].to(device) # printing indices not one got
            outputs = model(ids, mask, token_type_ids)  # pooler outputs of each sequence
            loss = loss_function(outputs, targets)
            tr_loss += loss.item()
            pred_big_val, pred_big_idx = torch.max(outputs.data, dim=1)  # big_idx is the location of max val found
            n_correct += (pred_big_idx==targets).sum().item()

            nb_tr_steps += 1
            nb_tr_examples += targets.size(0)

            if _ % 5 == 0 and _ != 0:
                loss_step = round(tr_loss / nb_tr_steps, 4)
                accu_step = round(n_correct / nb_tr_examples, 4)
                print(f"Training Loss per {_} steps: {loss_step}")
                print(f"Training Accuracy after {_} steps: {accu_step}")

            optimizer.zero_grad()  # clear out old gradients that already have been used to update weights
            loss.backward()  # calculate the gradient of loss
            # # When using GPU
            optimizer.step()  # update the parameters


        val_loss, preds, labels = valid(model,val_loader,loss_function)
        micro = f1_score(labels, preds, average='micro')
        macro = f1_score(labels, preds, average='macro')

        print(f"epoch {epoch}: validation loss = {val_loss},validation micro score: {micro}, validation macro score: {macro}")
        if early_stopper.early_stop(val_loss):
            print(f"early stopped at epoch {epoch}")
            epoch_loss = round(tr_loss / nb_tr_steps, 4)
            print(f'The average training accuracy after Epoch {epoch}: {(n_correct * 100) / nb_tr_examples}')
            print(f"Average training loss from {nb_tr_steps} steps: {epoch_loss}")

            return model, epoch_loss, optimizer
        #print out accuracies and loss after each epoch

    epoch_loss = round(tr_loss / nb_tr_steps, 4)
    print(f'The Total Accuracy after Epoch {epochs}: {(n_correct * 100) / nb_tr_examples}')
    print(f"Average training loss from {nb_tr_steps} steps: {epoch_loss}")

    return model, epoch_loss, optimizer

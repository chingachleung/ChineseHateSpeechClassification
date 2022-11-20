import torch
from tqdm import tqdm
from torch import cuda
from validation import valid
import numpy as np
device = 'cuda' if cuda.is_available() else 'cpu'
from sklearn.metrics import f1score
#https://stackoverflow.com/questions/71998978/early-stopping-in-pytorch
#https://github.com/christianversloot/machine-learning-articles/blob/main/how-to-use-k-fold-cross-validation-with-pytorch.md

#TRAIN_SIZE = 0.9
LEARNING_RATE = 1e-05

class EarlyStopper:
    def __init__(self, patience=3, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False



def train(model,epochs,training_loader, val_loader,onehot_encoder):
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
    early_stopper = EarlyStopper()
    f1_micro = F1Score(num_classes=3)
    f1_macro = F1Score(num_classes=3, average='macro')
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
            targets = data['targets'].to(device)
            target_big_val, target_big_idx = torch.max(targets, dim=1)
            outputs = model(ids, mask, token_type_ids)  # pooler outputs of each sequence
            loss = loss_function(outputs, target_big_idx)
            tr_loss += loss.item()
            pred_big_val, pred_big_idx = torch.max(outputs.data, dim=1)  # big_idx is the location of max val found
            #n_correct += calcuate_accuracy(pred_big_idx, target_big_idx)
            n_correct += (pred_big_idx==target_big_idx).sum().item()

            nb_tr_steps += 1
            nb_tr_examples += targets.size(0)

            if _ % 5 == 0 and _ != 0:
                loss_step = round(tr_loss / nb_tr_steps, 4)
                accu_step = round(n_correct / nb_tr_examples, 4)
                print(f"Training Loss per {_} steps: {loss_step}")
                print(f"Training Accuracy after {_} steps: {accu_step}")
                #loss_list.append(loss_step)
                #accuracy_list.append(accu_step)
            optimizer.zero_grad()  # clear out old gradients that already have been used to update weights
            loss.backward()  # calculate the gradient of loss
            # # When using GPU
            optimizer.step()  # update the parameters


        val_loss, preds, labels = valid(model,val_loader,onehot_encoder)
        micro = f1score(labels, average='micro')
        macro = f1score(labels, average='macro')

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

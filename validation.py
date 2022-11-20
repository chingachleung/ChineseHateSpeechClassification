import numpy as np
from tqdm import tqdm
import torch
from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'



def valid(model, validation_loader, onehot_encoder):
    model.eval()
    loss_function = torch.nn.CrossEntropyLoss()
    tr_loss = 0
    nb_tr_steps = 0
    predictions = np.array([])
    labels = np.array([])

    with torch.no_grad():
        for _, data in tqdm(enumerate(validation_loader, 0)):
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.long)
            target_big_val, target_big_idx = torch.max(targets, dim=1)
            targets = targets.cpu()
            inversed_targets = onehot_encoder.inverse_transform(targets)
            labels = np.append(labels, inversed_targets)
            outputs = model(ids, mask, token_type_ids)
            inversed_predictions = onehot_encoder.inverse_transform(outputs.data.tolist())
            predictions = np.append(predictions, inversed_predictions)
            #pred_big_val, pred_big_idx = torch.max(outputs.data, dim=1)
            loss = loss_function(outputs, target_big_idx)
            #n_correct += (pred_big_idx==target_big_idx).sum().item()
            tr_loss += loss.item()
            nb_tr_steps += 1
            #nb_tr_examples += targets.size(0)
    epoch_loss = tr_loss / nb_tr_steps
    return epoch_loss, predictions, labels

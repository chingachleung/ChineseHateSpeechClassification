"""
creating a Chinese BERT class
"""

import torch
from transformers import AutoModelForMaskedLM

class BertClass(torch.nn.Module):
    def __init__(self):
        super(BertClass, self).__init__()
        self.l1 = AutoModelForMaskedLM.from_pretrained("bert-base-chinese")
        #chinese BERT hidden dimensions
        self.pre_classifier = torch.nn.Linear(21128, 21128)
        self.dropout = torch.nn.Dropout(0.3)
        # three classes: hate-speech, abusive-only and neither
        self.classifier = torch.nn.Linear(21128, 3)

    def forward(self, input_ids, attention_mask, token_type_ids):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        hidden_state = output_1[0] # returning sequence outputs, if it's -1, it's all the hidden states
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output
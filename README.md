# ChineseHateSpeechClassification
Fine-tuning BERT-base-chinese model to train a model that classifies a tweet into hate-speech, abusive-only or neither. 

To start fine-tuning, run:

`python create_bert_model.py --train_file <train_file> --val_file <validation_file>`

To test a fine-tuned model on a test set, run:

`python test_model.py --testing_file <test_file.csv> --model_file <model.pt>`

This script prints out F1 scores (macro + micro),  and confusion matrix automatically. 

To create a further fine-tune a model that is already trained: 

`python transfer_learning.py --train_file <train_file.csv> --val_file <validation_file.csv> --model_file <model.pt>`

# Data 

Contains around 9,056 Tweets in Traditional Chinese, Simplified Chinse characters. Due to privacy restriction, I cannot publicly release the dataset. Please contact me directly at wl607@georgetown.edu to inquire about the data.




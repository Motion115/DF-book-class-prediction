from transformers import BertModel, BertConfig, BertTokenizer, AdamW, get_cosine_schedule_with_warmup

import torch
import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification

# Load the best model
loaded_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=24)
loaded_model.load_state_dict(torch.load('best_model.pth'))
loaded_model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
loaded_model.to(device)

# Load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

# Read data from test.csv
test_df = pd.read_csv("test.csv")

# Extract the 'Description' column
test_sentences = test_df["Description"].values

# Lists to store predictions
test_predictions = []

max_len=512
# Iterate through test sentences
for test_sentence in test_sentences:
    # Tokenize and encode the test sentence
    encoded_dict = tokenizer.encode_plus(
        test_sentence,
        add_special_tokens=True,
        max_length=max_len,
        pad_to_max_length=True,
        return_attention_mask=True,
        return_tensors='pt'
    )

    # Get input IDs and attention mask
    test_input_ids = encoded_dict['input_ids'].to(device)
    test_attention_mask = encoded_dict['attention_mask'].to(device)

    # Make predictions using the loaded model
    with torch.no_grad():
        outputs = loaded_model(test_input_ids, attention_mask=test_attention_mask)
        logits = outputs.logits
        predicted_label = np.argmax(logits.to('cpu').numpy(), axis=1)[0]
        test_predictions.append(predicted_label)

# Save predictions to a result.csv file, one label per line
result_df = pd.DataFrame({'Label': test_predictions})
result_df.to_csv('result.csv', index=False, header=False)



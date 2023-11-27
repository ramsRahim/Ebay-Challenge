import torch, argparse
import csv
import pandas as pd
from aspects import id_to_name
from tqdm import tqdm
from transformers import BertTokenizer, BertForTokenClassification
from transformers import AutoTokenizer, AutoModelForTokenClassification
from models import get_model_tokenizer

parser = argparse.ArgumentParser(description='NER Test')
parser.add_argument('--device', type=str, default='cuda:0', help='Device to run the model on')
parser.add_argument('--model_type', type=str, default='large', help='Model type: large or xlarge')  
args = parser.parse_args()

# Initialize the tokenizer and model
# tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
# model = BertForTokenClassification.from_pretrained('bert-base-multilingual-cased', num_labels=len(id_to_name))
# tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-german-cased")
# model = AutoModelForTokenClassification.from_pretrained("dbmdz/bert-base-german-cased", num_labels=len(id_to_name))
model, tokenizer = get_model_tokenizer(args.model_type, len(id_to_name))

model.to(args.device)
# state = torch.load('saved_models/xlm_roberta_large_again.pt')
state = torch.load('saved_models/xlm_roberta_large_eval_new_fold_0.pt')
model.load_state_dict(state['model_state_dict'])
model.eval()
print('best f1:', state['best_f1'])

# Load data
df = pd.read_csv('data/test_set.tsv', delimiter='\t', quoting=csv.QUOTE_NONE)
sentences = df.Title.values
all_records = df['Record Number']

# Set batch size
batch_size = 32  # You can adjust this based on your GPU's memory
output_data = []

# Batch-wise prediction
for i in tqdm(range(0, len(sentences), batch_size), total=len(sentences) // batch_size + 1):
    batch_sentences = sentences[i:i + batch_size].tolist()
    batch_records = all_records[i:i + batch_size]

    # Tokenize the batch
    encoded_inputs = tokenizer(batch_sentences, padding=True, truncation=True, return_tensors="pt")
    input_ids = encoded_inputs['input_ids'].to(args.device)
    attention_masks = encoded_inputs['attention_mask'].to(args.device)
    
    # Predict
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_masks)
    batch_predictions = outputs.logits.argmax(dim=2).cpu().numpy()

    # Process each sentence in the batch
    for j, (sentence, predictions) in enumerate(zip(batch_sentences, batch_predictions)):
        words = sentence.split()
        predictions = predictions.tolist()[1:-1]  # Remove CLS and SEP tokens

        aspect_names = []
        aspect_values = []

        for word in words:
            tokens = tokenizer.tokenize(word)
            word_label = id_to_name[int(predictions[0])]
            predictions = predictions[len(tokens):]

            if word_label in ['No Tag', 'Obscure']:
                continue

            if word_label.endswith('-I') and aspect_values:
                aspect_values[-1] = aspect_values[-1] + ' ' + word
            else:
                aspect_values.append(word)
                if word_label.endswith('-I'):
                    word_label = word_label[:-2]
                aspect_names.append(word_label)

        record_numbers = [batch_records.iloc[j]] * len(aspect_names)
        output_data.extend(zip(record_numbers, aspect_names, aspect_values))

# Write to a TSV file
with open('submissions/xlm_roberta_large_eval_new_fold_0.tsv', 'wt', newline='', encoding='utf-8') as f:
    tsv_writer = csv.writer(f, delimiter='\t')
    for line in output_data:
        tsv_writer.writerow(line)

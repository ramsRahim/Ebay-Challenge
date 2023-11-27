import torch
import csv, argparse
import pandas as pd
from aspects import id_to_name
from tqdm import tqdm
from transformers import BertTokenizer, BertForTokenClassification

parser = argparse.ArgumentParser(description='NER Test')
parser.add_argument('--device', type=str, default='cuda:0', help='Device to run the model on')
args = parser.parse_args()


df = pd.read_csv('data/test_set.tsv', delimiter='\t', quoting=csv.QUOTE_NONE)
print(df.head())

print(len(df))

# Initialize the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')


# Load the tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
model = BertForTokenClassification.from_pretrained('bert-base-multilingual-cased', num_labels=len(id_to_name))
model.to(args.device)
model.load_state_dict(torch.load('saved_models/model.pt')['model_state_dict'])
model.eval()

sentences = df.Title.values
all_records = df['Record Number']


output_data = []

for idx, sentence in tqdm(enumerate(sentences), total=len(sentences)):
    words = sentence.split()

    with torch.no_grad():
        encoded_input = tokenizer.encode_plus(sentence, return_tensors="pt")
        input_ids = encoded_input['input_ids'].to(args.device)
        
        attention_masks = torch.ones(input_ids.shape, dtype=torch.long).to(args.device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_masks)
        predictions = outputs.logits.argmax(dim=2).cpu().numpy()
        
    predictions = predictions[0].tolist()[1:-1]  # Remove CLS and SEP tokens
        
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
            
    assert len(aspect_names) == len(aspect_values)
    assert len(predictions) == 0
            
    # Store the results for each sentence
    record_numbers = [all_records[idx]] * len(aspect_names)
    output_data.extend(zip(record_numbers, aspect_names, aspect_values))
    
    # if idx == 100:
    #     break

# Write to a TSV file
with open('submissions/output.tsv', 'wt', newline='', encoding='utf-8') as f:
    tsv_writer = csv.writer(f, delimiter='\t')
    # tsv_writer.writerow(['Record Number', 'Aspect Name', 'Aspect Value'])  # Header
    for line in output_data:
        tsv_writer.writerow(line)

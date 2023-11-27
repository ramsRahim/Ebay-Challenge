import csv
import pandas as pd
# from transformers import BertTokenizer
from transformers import BertTokenizerFast
from tqdm import tqdm
from aspects import name_to_id, id_to_name

tokenizer = BertTokenizerFast.from_pretrained('bert-base-multilingual-cased')



# def tokenize_and_align_labels(all_tokens, all_labels):  
#     tokenized_inputs = tokenizer(all_tokens, truncation=True, is_split_into_words=True)

#     labels = []
#     for i, label in enumerate(all_labels):
#         word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
#         print(tokenized_inputs)
        
#         print(len(word_ids), len(label))
#         previous_word_idx = None
#         label_ids = []
#         for word_idx in word_ids:  # Set the special tokens to -100.
#             if word_idx is None:
#                 label_ids.append(-100)
#             elif word_idx != previous_word_idx:  # Only label the first token of a given word.
#                 label_ids.append(label[word_idx])
#             else:
#                 label_ids.append(-100)
#             previous_word_idx = word_idx
#         labels.append(label_ids)

#     tokenized_inputs["labels"] = labels
#     return tokenized_inputs


df = pd.read_csv('data/Train_Tagged_Titles.tsv', delimiter='\t', quoting=csv.QUOTE_NONE)
df['Tag'] = df['Tag'].fillna(method='ffill')

grouped_df = df.groupby("Record Number")

all_tokens = []
all_labels = []
for record in tqdm(grouped_df,total=len(grouped_df)):
    record = record[1]
    # print(record)
    words = record.Token.values
    tags = record.Tag.values
    
    tokens = [tokenizer.cls_token_id]
    labels = [-100]
    
    for word, tag in zip(words, tags):
        input_ids = tokenizer.encode(word, add_special_tokens=False)
        tokens.extend(input_ids)
        labels.append(name_to_id[tag])
        if len(input_ids) > 1:
            labels.extend([-100] * (len(input_ids) - 1))
            
    tokens.append(tokenizer.sep_token_id)
    labels.append(-100)
        
    # all_sentences.append(words)
    
    all_tokens.append(tokens)
    all_labels.append(labels)
    

def decode_labels(tokens, labels, tokenizer):
    decoded_labels = []
    for token_id, label in zip(tokens, labels):
        if label != -100:
            # Convert the token ID to the token string
            token = tokenizer.convert_ids_to_tokens(token_id)
            decoded_label = id_to_name[label]  # Assuming you have a mapping from ID to label name
            decoded_labels.append((token, decoded_label))
    return decoded_labels

# Example usage
decoded_labels = decode_labels(all_tokens[0], all_labels[0], tokenizer)
print(decoded_labels)

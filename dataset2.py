import csv
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
from transformers import BertTokenizer
from tqdm import tqdm
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from transformers import AutoTokenizer
from models import get_tokenizer


# Assuming 'name_to_id' maps from NER tags to IDs
from aspects import name_to_id

class NERDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def align_labels(sentence, tags):

    # Tokenize the sentence
    encoded_input = tokenizer.encode_plus(sentence, return_tensors="pt")
    input_ids = encoded_input['input_ids'][0]

    # Tokenize words individually to count subtokens
    tokenized_words = [tokenizer.tokenize(word) for word in sentence.split()]
    assert len(tokenized_words) == len(tags)

    # Align labels
    aligned_labels = [-100]
    for word, tag in zip(tokenized_words, tags):
        # First subtoken gets the original label, rest get -100
        word_labels = [name_to_id[tag]] + [-100] * (len(word) - 1)
        aligned_labels.extend(word_labels)

    aligned_labels.append(-100)
    return input_ids, aligned_labels


def collate_fn(batch):
    """
    Collate function to pad input sequences for batch processing.

    Args:
        batch (list of tuples): A list where each tuple contains input_ids and labels.

    Returns:
        dict: A dictionary with padded 'input_ids' and 'labels', along with 'attention_mask'.
    """

    input_ids = [torch.tensor(data['input_ids'], dtype=torch.long) for data in batch]
    labels = [torch.tensor(data['labels'], dtype=torch.long) for data in batch]
    
    # Padding input_ids and labels
    # Note: pad_sequence pads with 0 by default, which is common for most models
    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100)  # Assuming -100 is the ignore index for labels

    # Create attention masks
    # Mask is 1 for tokens and 0 for padding
    attention_masks = torch.tensor([[float(i != tokenizer.pad_token_id) for i in ii] for ii in input_ids_padded])

    return {
        'input_ids': input_ids_padded,
        'attention_mask': attention_masks,
        'labels': labels_padded
    }
    
    
def forward_fill(df):
    # Forward fill using a for loop with '-I' suffix
    last_tag = None
    for index, row in df.iterrows():
        if pd.isna(row['Tag']) and last_tag is not None:
            # print(row['Tag'])
            if last_tag not in ['No Tag', 'Obscure']:
                df.at[index, 'Tag'] = last_tag + '-I'  
            else:
                df.at[index, 'Tag'] = last_tag 
            
        elif not pd.isna(row['Tag']):
            last_tag = row['Tag']
            # if last_tag != 'O':  # Don't append '-I' if the tag is 'O'
            #     last_tag += '-I'
            #     df.at[index, 'Tag'] = last_tag


def get_dataloaders(batch_size=32, fold=0, model_type='large', n_splits=5):

    global tokenizer

    # Initialize the tokenizer
    # tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-german-cased")

    # tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    tokenizer = get_tokenizer(model_type)

    df = pd.read_csv('data/Train_Tagged_Titles.tsv', delimiter='\t', quoting=csv.QUOTE_NONE)
    # df['Tag'] = df['Tag'].fillna(method='ffill')
    # df['Tag'] = df['Tag'].ffill().apply(lambda x: x if x == 'O' else x + '-I')
    forward_fill(df)
    
    # df['Tag'] = df['Tag'].fillna('Empty')
    # print(df['Tag'].value_counts())

    grouped_df = df.groupby("Record Number")

    # Prepare data for NER dataset
    all_input_ids = []
    all_labels = []
    one_hot_labels = []
    for record in tqdm(grouped_df, total=len(grouped_df)):
        record_data = record[1]
        sentence = record_data.Title.values[0]
        tags = record_data.Tag.values

        input_ids, labels = align_labels(sentence, tags)
        one_hot_label = [0] * len(name_to_id)
        for label in labels:
            if label != -100:
                one_hot_label[label] = 1

        # Add to dataset lists
        all_input_ids.append(input_ids.numpy())
        all_labels.append(labels)
        one_hot_labels.append(one_hot_label)

    # K-Fold Splitting
    mskf = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    folds = mskf.split(all_input_ids, one_hot_labels)

    for cur_fold, (train_index, test_index) in enumerate(mskf.split(all_input_ids, one_hot_labels)):
        if cur_fold != fold:
            continue

        train_ids = [all_input_ids[i] for i in train_index]   
        train_labels = [all_labels[i] for i in train_index]

        test_ids = [all_input_ids[i] for i in test_index]
        test_labels = [all_labels[i] for i in test_index]

        # Create the dataset
        train_encodings = {'input_ids': train_ids}
        train_dataset = NERDataset(train_encodings, train_labels)

        test_encodings = {'input_ids': test_ids}
        test_dataset = NERDataset(test_encodings, test_labels)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    return train_loader, test_loader




# print(ner_dataset[0])

# # Example DataLoader usage
# data_loader = DataLoader(ner_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)


if __name__ == '__main__':
    train_loader, _ = get_dataloaders()
    # for batch in train_loader:
    #     print(batch)
    #     break


    

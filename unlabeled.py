import torch
import csv
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from models import get_tokenizer
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

df = pd.read_csv('data/Listing_Titles.tsv', delimiter='\t', quoting=csv.QUOTE_NONE)

sub_df = df[df['Record Number'] > 30000]
print(len(sub_df))

class NERDatasetUnlabeled(Dataset):
    def __init__(self, sentences, tokenizer):
        self.sentences = sentences
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        encoded_input = self.tokenizer.encode_plus(sentence, return_tensors="pt")
        input_ids = encoded_input['input_ids'][0]
        return input_ids

    def __len__(self):
        return len(self.sentences)


def collate_fn(batch):
    """
    Collate function to pad input sequences for batch processing.

    Args:
        batch (list of tuples): A list where each tuple contains input_ids and labels.

    Returns:
        dict: A dictionary with padded 'input_ids' and 'labels', along with 'attention_mask'.
    """
    input_ids = pad_sequence(batch, batch_first=True, padding_value=1)
    attention_mask = input_ids != 0
    return {'input_ids': input_ids, 'attention_mask': attention_mask}

def get_unlabel_dataloader(batch_size, tokenizer):
    dataset = NERDatasetUnlabeled(sub_df.Title.values, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    return dataloader

@torch.no_grad()
def generate_pseudo_labels(dataloader, data_iter, model, tokenizer, num_steps, device):

    cur_state_dict = model.state_dict()
    best_state_dict = torch.load('saved_models/xlm_roberta_large_semi_baseline.pt')['model_state_dict']
    # load the best state dict
    model.load_state_dict(best_state_dict)
    model.eval()

    sep_token_id = tokenizer.sep_token_id
    pad_token_id = tokenizer.pad_token_id	
    assert pad_token_id == 1, "Padding ID should be 1, but got {}".format(pad_token_id)

    data = []
    
    for step in tqdm(range(num_steps)):
        try:
            batch = next(data_iter)
        except StopIteration:
            # Reinitialize the iterator if the dataset ends
            data_iter = iter(dataloader)
            batch = next(data_iter)

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        predictions = outputs.logits.argmax(dim=2)
        
        # Set [CLS] and [SEP] tokens to -100
        # sep_token_id = 2  # Adjust if using a different model
        predictions[:, 0] = -100  # [CLS] tokens at the start
        sep_indices = (input_ids == sep_token_id).nonzero(as_tuple=True)
        pad_indices = (input_ids == pad_token_id).nonzero(as_tuple=True)

        predictions[sep_indices] = -100  # [SEP] tokens
        predictions[pad_indices] = -100
        data.append((input_ids.cpu(), attention_mask.cpu(), predictions.cpu()))
    # load back the original state dict
    model.load_state_dict(cur_state_dict)
    return data

# sentences

# encoded_inputs = tokenizer(batch_sentences, padding=True, truncation=True, return_tensors="pt")
# input_ids = encoded_inputs['input_ids'].to(args.device)
# attention_masks = encoded_inputs['attention_mask'].to(args.device)
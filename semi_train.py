import argparse
import torch
import warnings
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForTokenClassification, AdamW
from utils import seed_everything, evaluate
from tqdm import tqdm
from aspects import name_to_id
from dataset2 import get_dataloaders
from torch.cuda.amp import GradScaler, autocast
from transformers import get_linear_schedule_with_warmup
from transformers import AutoTokenizer, AutoModelForTokenClassification
from models import get_model_tokenizer
from unlabeled import get_unlabel_dataloader, generate_pseudo_labels
from torch.nn.utils.rnn import pad_sequence


warnings.filterwarnings('ignore')

# Create an argument parser
parser = argparse.ArgumentParser(description='NER Training')

# Add arguments
# parser.add_argument('--data_path', type=str, default='/path/to/training/data', help='Path to training data')
# parser.add_argument('--model_path', type=str, default='/path/to/save/model', help='Path to save the trained model')
# parser.add_argument('--num_labels', type=int, default=10, help='Number of labels')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs')
parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
parser.add_argument('--device', type=str, default='cuda:0', help='Device to run the model on')
parser.add_argument('--mixed_precision', action='store_true', help='Use mixed precision FP16')
parser.add_argument('--lr_scheduler', action='store_true', help='Use learning rate scheduler')
parser.add_argument('--model_type', type=str, default='large', help='Model type: large or xlarge')

# Parse the arguments
args = parser.parse_args()

seed_everything()

# Set the device (GPU or CPU)
device = torch.device(args.device)

# Load the tokenizer and model
# tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
# model = BertForTokenClassification.from_pretrained('bert-base-multilingual-cased', num_labels=len(name_to_id))
model, tokenizer = get_model_tokenizer(args.model_type, len(name_to_id))

# tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
# tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-german-cased")
# model = AutoModelForTokenClassification.from_pretrained("dbmdz/bert-base-german-cased", num_labels=len(name_to_id))

# Set the model to the device
model.to(device)

# Load the training data

train_dataloader, test_dataloader = get_dataloaders(batch_size=args.batch_size, fold=0, model_type=args.model_type)

# Set the optimizer and learning rate
optimizer = AdamW(model.parameters(), lr=args.lr)
best_f1 = 0

scaler = GradScaler()
if args.mixed_precision:
    print('Mixed precision training enabled...')
    
if args.lr_scheduler:
    # Total number of training steps
    total_steps = len(train_dataloader) * args.num_epochs

    # Define the scheduler
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps=int(total_steps * 0.1),  # Default value in run_glue.py
                                                num_training_steps=total_steps)    


unlabeled_loader = get_unlabel_dataloader(args.batch_size, tokenizer)
data_iter = iter(unlabeled_loader)
pseudo_data = None

# Training loop
for epoch in range(args.num_epochs):
    pbar = tqdm(train_dataloader, total=len(train_dataloader))
    model.train()
    for b_idx, batch in enumerate(pbar):
        input_ids, attention_mask, batch_labels = batch['input_ids'], batch['attention_mask'], batch['labels']
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        batch_labels = batch_labels.to(device)


        # Forward pass and compute loss for original data
        optimizer.zero_grad()
        with autocast(enabled=args.mixed_precision):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=batch_labels)
            original_loss = outputs.loss

        # If pseudo data is available, perform a separate forward pass
        if pseudo_data is not None:
            input_ids_pseudo, attention_mask_pseudo, batch_labels_pseudo = pseudo_data[b_idx]
            input_ids_pseudo = input_ids_pseudo.to(device)
            attention_mask_pseudo = attention_mask_pseudo.to(device)
            batch_labels_pseudo = batch_labels_pseudo.to(device)

            with autocast(enabled=args.mixed_precision):
                outputs_pseudo = model(input_ids=input_ids_pseudo, attention_mask=attention_mask_pseudo, labels=batch_labels_pseudo)
                pseudo_loss = outputs_pseudo.loss

            # Combine the losses (you can also average them if preferred)
            total_loss = original_loss + pseudo_loss
        else:
            total_loss = original_loss

        pbar.set_postfix({'loss': total_loss.item()})

        # Scales loss. Calls backward() on scaled loss to create scaled gradients.
        scaler.scale(total_loss).backward()

        # Updates the weights.
        scaler.step(optimizer)
        
        if args.lr_scheduler:
            scheduler.step()

        # Updates the scale for next iteration.
        scaler.update()

    # Evaluate at the end of each epoch
    eval_f1 = evaluate(model, test_dataloader, device)
    if eval_f1 > best_f1:
        best_f1 = eval_f1
        state_dict = {'model_state_dict':model.state_dict(), 'best_f1': best_f1}
        torch.save(state_dict, 'saved_models/xlm_roberta_large_semi_baseline.pt')
        print('validation f1 improved, saving model...')

    if epoch >= 20:
        pseudo_data = generate_pseudo_labels(unlabeled_loader, data_iter, model, tokenizer, len(train_dataloader), device)
    
    print(f'Epoch {epoch + 1} - F1: {eval_f1:.3f}')



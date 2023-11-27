import os
import torch
import random
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, f1_score

def seed_everything(seed=42):
    """
    Seed everything to make the code more deterministic.

    Args:
        seed (int): The seed number. Default is 42.
    """
    random.seed(seed)  # Python random module
    np.random.seed(seed)  # Numpy module
    os.environ["PYTHONHASHSEED"] = str(seed)  # Python environment
    
    torch.manual_seed(seed)  # PyTorch
    torch.cuda.manual_seed(seed)  # PyTorch CUDA (for deterministic GPU operations)
    torch.cuda.manual_seed_all(seed)  # If using multi-GPU
    torch.backends.cudnn.deterministic = True  # CuDNN optimizations
    torch.backends.cudnn.benchmark = False



def evaluate(model, test_dataloader, device):
    model.eval()  # Set the model to evaluation mode

    true_labels = []
    pred_labels = []
    
    with torch.no_grad():  # No need to track gradients during evaluation
        for batch in test_dataloader:
            # Move batch to the device we are using
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass, get predictions
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = labels.to('cpu').numpy()



            # Convert logits to actual predictions
            predictions = np.argmax(logits, axis=2)

            # Flatten the batch for true labels and predictions
            # and ignore special tokens (usually -100)
            true_label_list = []
            pred_label_list = []

            for i in range(label_ids.shape[0]):
                true_label_list.extend([label for label in label_ids[i] if label != -100])
                pred_label_list.extend([pred for label, pred in zip(label_ids[i], predictions[i]) if label != -100])

                assert len(true_label_list) == len(pred_label_list)

            # true_label_list = [label for sublist in label_ids for label in sublist if label != -100]
            # pred_label_list = [pred for sublist in predictions for label, pred in zip(sublist, sublist) if label != -100]
            
            true_labels.extend(true_label_list)
            pred_labels.extend(pred_label_list)

    # Compute metrics
    # _, _, f1_weighted, _ = precision_recall_fscore_support(true_labels, pred_labels, average='weighted')
    f1_weighted = f1_score(true_labels, pred_labels, average='weighted')
    return f1_weighted

def calculate_weighted_f1(true_labels, pred_labels):
    # Calculate F1 for each class individually
    # f1_per_class = f1_score(true_labels, pred_labels, average=None)[2:] # Ignore the first two classes
        # Calculate F1 for each class individually
    f1_per_class = f1_score(true_labels, pred_labels, average=None, labels=range(2,70))#[:2] # Ignore the first two classes


    # Count the number of instances for each class
    class_counts = np.bincount(true_labels)[2:] # Ignore the first two classes

    # Calculate the fraction of instances per class
    class_weights = class_counts / np.sum(class_counts)

    # Compute the weighted average of F1 scores
    f1_weighted = np.sum(f1_per_class * class_weights)

    return f1_weighted


def evaluate_new(model, test_dataloader, device):
    model.eval()  # Set the model to evaluation mode

    true_labels = []
    pred_labels = []
    
    with torch.no_grad():  # No need to track gradients during evaluation
        for batch in test_dataloader:
            # Move batch to the device we are using
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass, get predictions
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = labels.to('cpu').numpy()



            # Convert logits to actual predictions
            predictions = np.argmax(logits, axis=2)

            # Flatten the batch for true labels and predictions
            # and ignore special tokens (usually -100)
            true_label_list = []
            pred_label_list = []

            for i in range(label_ids.shape[0]):
                true_label_list.extend([label for label in label_ids[i] if label != -100])
                pred_label_list.extend([pred for label, pred in zip(label_ids[i], predictions[i]) if label != -100])

                assert len(true_label_list) == len(pred_label_list)

            # true_label_list = [label for sublist in label_ids for label in sublist if label != -100]
            # pred_label_list = [pred for sublist in predictions for label, pred in zip(sublist, sublist) if label != -100]
            
            true_labels.extend(true_label_list)
            pred_labels.extend(pred_label_list)

    # Compute metrics
    # _, _, f1_weighted, _ = precision_recall_fscore_support(true_labels, pred_labels, average='weighted')
    f1_weighted = calculate_weighted_f1(true_labels, pred_labels)
    
    return f1_weighted


# Example usage
# f1_weighted = evaluate_weighted_f1(model, test_dataloader, device)


# Example usage
# precision, recall, f1 = evaluate(model, test_dataloader, device)


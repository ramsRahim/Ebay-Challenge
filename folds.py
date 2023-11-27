import pandas as pd
import numpy as np
import csv
from collections import Counter
from skmultilearn.model_selection import IterativeStratification
import matplotlib.pyplot as plt

# Load and preprocess data
df = pd.read_csv('data/Train_Tagged_Titles.tsv', delimiter='\t', quoting=csv.QUOTE_NONE)
df['Tag'] = df['Tag'].fillna(method='ffill')
grouped_df = df.groupby("Record Number").apply(lambda s: [(w, t) for w, t in zip(s["Token"], s["Tag"])])

# Extract unique labels for each record
record_labels = df.groupby("Record Number")['Tag'].unique()

# Multi-hot encoding of labels
unique_tags = df['Tag'].unique()
tag_to_index = {tag: i for i, tag in enumerate(unique_tags)}
multi_hot_encoded_labels = []

for labels in record_labels:
    encoded = [0] * len(unique_tags)
    for label in labels:
        encoded[tag_to_index[label]] = 1
    multi_hot_encoded_labels.append(encoded)

# Convert to appropriate format for stratification
X = np.arange(len(grouped_df))
y = np.array(multi_hot_encoded_labels)

# Create Stratified K-Folds
stratifier = IterativeStratification(n_splits=5, order=1)
folds = stratifier.split(X, y)

# Specify the tags of interest
tags_of_interest = ['No Tag', 'Modell', 'Produktart', 'Marke', 'Farbe', 
                    'Abteilung', 'Produktlinie', 'Stil', 'Herstellernummer', 'EU-Schuhgröße']

# Function to calculate tag distribution for specified tags
def get_tag_distribution(indices, df, tags_of_interest):
    tags = [tag for idx in indices for _, tag in df.iloc[idx]]
    tag_counter = Counter(tags)
    return {tag: tag_counter[tag] for tag in tags_of_interest}

# Analyze and store tag distribution per fold
train_distributions = []
test_distributions = []

for fold_idx, (train_idx, test_idx) in enumerate(folds):
    train_dist = get_tag_distribution(train_idx, grouped_df, tags_of_interest)
    test_dist = get_tag_distribution(test_idx, grouped_df, tags_of_interest)
    train_distributions.append(train_dist)
    test_distributions.append(test_dist)

# Visualizing tag distribution
fig, axes = plt.subplots(2, 5, figsize=(20, 8), sharey=True)
fig.suptitle('Tag Distribution per Fold')

# Plotting for each fold
for i in range(5):
    axes[0, i].bar(train_distributions[i].keys(), train_distributions[i].values())
    axes[0, i].set_title(f'Train Fold {i + 1}')
    axes[0, i].tick_params(labelrotation=90)

    axes[1, i].bar(test_distributions[i].keys(), test_distributions[i].values())
    axes[1, i].set_title(f'Test Fold {i + 1}')
    axes[1, i].tick_params(labelrotation=90)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('plots/tag_distribution.png')

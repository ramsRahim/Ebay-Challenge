import csv
import pandas as pd
# from transformers import BertTokenizer
from transformers import BertTokenizerFast
from tqdm import tqdm


df = pd.read_csv('data/Train_Tagged_Titles.tsv', delimiter='\t', quoting=csv.QUOTE_NONE)
df['Tag'] = df['Tag'].fillna(method='ffill')

grouped_df = df.groupby("Record Number").apply(lambda s: [(w, t) for w, t in zip(s["Token"], s["Tag"])])

tokenizer = BertTokenizerFast.from_pretrained('bert-base-multilingual-cased')

tokenized_inputs = []
tags = []

# max_length = 0
# pbar = tqdm(grouped_df, total=len(grouped_df))
# for record in pbar:
#     words = [pair[0] for pair in record]
#     labels = [pair[1] for pair in record]
#     assert len(words) == len(labels)

    # Example data
    # words = ['Hello', 'world']
    # labels = ['O', 'B-LOC']  # Assuming "world" is labeled as a location

    # Initialize tokenizer
    # tokenizer = BertTokenizerFast.from_pretrained('bert-base-multilingual-cased')

    # Tokenize with offset mapping
    # encoding = tokenizer(words, is_split_into_words=True, return_offsets_mapping=True)
    # offset_mapping = encoding['offset_mapping']

    # print("Tokens:", encoding.tokens())
    # print("Offset Mapping:", offset_mapping)

    # Align labels with subword tokens
    aligned_labels = []
    word_index = 0  # Index of the word in the original list

    # for offset in offset_mapping:
    #     # Check for special tokens
    #     if offset == (0, 0):
    #         aligned_labels.append('O')  # Assign 'O' for special tokens
    #     else:
    #         # Assign label of the original word to the subword token
    #         if word_index < len(words) and offset[1] <= len(words[word_index]):
    #             aligned_labels.append(labels[word_index])
    #         else:
    #             aligned_labels.append('X')  # If index goes beyond, assign 'O'

    #         # Move to the next word when the end of the current word is reached
    #         if word_index < len(words) and offset[1] == len(words[word_index]):
    #             word_index += 1

    # for i, offset in enumerate(offset_mapping):
    #     # Check for special tokens
    #     if offset == (0, 0):
    #         aligned_labels.append('O')  # Assign 'O' for special tokens
    #     else:
    #         # Assign label of the original word to the first subword token
    #         if offset[0] == 0:
    #             aligned_labels.append(labels[word_index])
    #         else:
    #             aligned_labels.append('X')  # Assign 'X' for subsequent subword tokens

    #         # Move to the next word when the end of the current word is reached
    #         if word_index < len(words) and offset[1] == len(words[word_index]):
    #             word_index += 1

    # # max_length = max(max_length, len(encoding['input_ids']))
    # # pbar.set_description(f"max_length: {max_length}")
    # # print(max_length)
    # print("Tokens:", encoding.tokens())
    # print("Aligned Labels:", aligned_labels)
    # break
# print('max_length:', max_length)    

# Decoding the labels to match original labels
    # original_labels = []
    # for i, token in enumerate(encoding.tokens()):
    #     if aligned_labels[i] != 'X':
    #         original_labels.append((token, aligned_labels[i]))

    # print("Original Words and Labels:", list(zip(words, labels)))
    # print("Decoded Tokens and Labels:", original_labels)
    # assert len(words) == len(original_labels[1:-1])


# for record in grouped_df:
#     word_list = [pair[0] for pair in record]
#     tag_list = [pair[1] for pair in record]
#     print(word_list)
#     print(tag_list)

#     bert_input = tokenizer(word_list, is_split_into_words=True, return_offsets_mapping=True, padding='max_length', truncation=True)
    
#     # print(word_list)
#     # Initialize a list for the new tags
#     new_tags = ['O'] * len(bert_input['input_ids'])

#     # Iterate through the offset mappings
#     word_idx = 0
#     for idx, offset in enumerate(bert_input.offset_mapping):
#         print(offset)
#         # Only update tag if it is the start of a word
#         if offset[0] == 0 and word_idx < len(tag_list):
#             new_tags[idx] = tag_list[word_idx]
#             word_idx += 1

#     tokenized_inputs.append(bert_input)
#     tags.append(new_tags)
#     break

# print(tokenized_inputs[0]['input_ids'][:23])
# print(tags[0])

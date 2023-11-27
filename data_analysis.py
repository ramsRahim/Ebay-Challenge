import csv
import pandas as pd
from aspects import *


# df = pd.read_tsv('data/Train_Tagged_Titles.tsv')

# df = pd.read_csv('data/Train_Tagged_Titles.tsv', delimiter='\t', na_values=['?'])

# print(df.head())

df = pd.read_csv('data/Train_Tagged_Titles.tsv', delimiter='\t', quoting=csv.QUOTE_NONE)
print(df.head(16))

# Assuming df is your DataFrame
df['Tag'] = df['Tag'].fillna(method='ffill')
print(df.head(16))

# print(df.Tag.value_counts())
# print(type(df.Tag.value_counts()))

for value, count in df.Tag.value_counts().items():
    print(value, '----', aspect_names[value], '---', count)

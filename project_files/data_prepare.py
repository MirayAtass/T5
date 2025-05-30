import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset
import re

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zçğöşüı\s]', '', text)
    text = ' '.join(text.split())
    
    return text

train_df = train_df[['text', 'label']].dropna()
test_df = test_df[['text', 'label']].dropna()

train_df['text'] = train_df['text'].apply(clean_text)
test_df['text'] = test_df['text'].apply(clean_text)

def balance_dataset(df, n_per_class):
    return df.groupby("label").sample(n=n_per_class, random_state=42)

train_balanced = balance_dataset(train_df, n_per_class=90)
test_balanced = balance_dataset(test_df, n_per_class=30)

train_df, val_df = train_test_split(
    train_balanced, test_size=0.1, stratify=train_balanced['label'], random_state=42)

train_df.to_csv("train_balanced.csv", index=False)
val_df.to_csv("val_balanced.csv", index=False)
test_balanced.to_csv("test_balanced.csv", index=False)

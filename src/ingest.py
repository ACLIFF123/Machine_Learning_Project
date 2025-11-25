import json
import os
import csv
import pandas as pd


RAW_DATA_PATH = "news_data/raw_data/ag_news.jsonl"
CLEANED_DATA_PATH = "news_data/clean_data/ag_news_clean.csv"


LABEL_MAP = {
    1: "world",
    2: "sports",
    3: "business",
    4: "sci/tech",
}


# Loads jsonl 
def load_data(filepath):
    return pd.read_json(filepath, lines=True)


def clean_data(df):
    # Combine title + description into one text field
    df["news_text"] = df["title"] + " " + df["description"]

    # Convert numeric label → string label
    df["category"] = df["label"].map(LABEL_MAP)

    # Keep only the cleaned columns
    df = df[["news_text", "category"]]

    # Remove any rows with missing values
    df = df.dropna()

    return df                   


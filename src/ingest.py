import json
import os
import csv
import pandas as pd


RAW_DATA_PATH = "news_data/raw_data/ag_news.jsonl"
CLEANED_DATA_PATH = "news_data/clean_data/ag_news_clean.csv"


# Loads jsonl 
def load_data(filepath):
    return pd.read_json(filepath, lines=True)


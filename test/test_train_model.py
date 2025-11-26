from src.train_model import load_cleaned_data
import pandas as pd
import os 



def test_load_dataset(tmp_path):

    
    csv_file = tmp_path / "sample.csv"
    csv_file.write_text("news_text,category\n", encoding="utf-8")

    # Call the function
    df = load_cleaned_data(csv_file)

    assert isinstance(df, pd.DataFrame)
    assert "news_text" in df.columns
    assert "category" in df.columns
    


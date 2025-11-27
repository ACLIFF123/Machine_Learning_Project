from src.interface.news_classifier import NewsClassifier
from src.user_interface import ui_predict



def test_output_lables_ui():

    result = ui_predict("stock market boom")
    
    assert result == "business"




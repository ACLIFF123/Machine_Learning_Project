from src.interface.news_classifier import NewsClassifier
from src.logger import get_logger

logger = get_logger(__name__)


def ui_predict(text: str):
    classifier = NewsClassifier("model/model.pkl")
    return classifier.classify(text)

def main():

    print("===================================")
    print("Welcome to Your News Categry Identifier")
    print("===================================\n")

    classifier = NewsClassifier("model/model.pkl")

    while True:
        # Get user input
        user_input = input("Enter News Title: ").strip()

        # Exit condition
        if user_input.lower() in ["quit", "exit"]:
            print("\n Goodbye!")
            break

        if not user_input:
            print("Please enter some text im am only here to help.")
        try:
            prediction = classifier.classify(user_input)
            print(f"[Result] {prediction}\n")

        except Exception as e:
            logger.error(e)
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
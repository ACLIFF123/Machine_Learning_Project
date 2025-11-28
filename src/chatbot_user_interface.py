from src.interface.news_classifier import NewsClassifier
from src.logger import get_logger
from src.models.extraction import ExtractorLLM

logger = get_logger(__name__)

extractor_llm = ExtractorLLM()

classifier = NewsClassifier("model/model.pkl")


def ui_process(text: str) -> tuple[str, str, str]:

    classifier = NewsClassifier("model/model.pkl")

    headline = extractor_llm.extract(text)

    category = classifier.classify(headline)

    return headline, category

def main():
    print("===================================")
    print("Welcome to Your News Categry Identifier")
    print("===================================\n")

    classifier = NewsClassifier("model/model.pkl")

    while True:
        user_input = input("Enter News Title: ").strip()

        if user_input.lower() in ["quit", "exit"]:
            print("\n Goodbye!")
            break

        if not user_input:
            print("Please enter some text im am only here to help.")

        try:
        
            headline, category = ui_process(user_input)

            print(f"\n[Extracted Headline] {headline}")


            print(f"[Category] {category}")

            
        except Exception as e:
            logger.error(e)
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
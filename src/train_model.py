from models.data_loader import load_cleaned_data, split_dataset
from models.model_classifier import ClassificationModel
from logger import get_logger


logger = get_logger(__name__)



def main():
    logger.info("Loading cleaned dataset")
    df = load_cleaned_data()

    
    X = df["news_text"]
    y = df["category"]

    logger.info("Splitting dataset")
    X_train, X_val, y_train, y_val = split_dataset(X, y)

    # Print sizes
    logger.info(f"Train size: {len(X_train)}")
    logger.info(f"Validation size: {len(X_val)}")

    # Print examples
    logger.debug("\nExample train row:")
    logger.debug(f"{X_train.iloc[0]} → {y_train.iloc[0]}")

    logger.debug("\nExample validation row:")
    logger.debug(f"{X_val.iloc[0]} → {y_val.iloc[0]}")


    model = ClassificationModel(model_type="logistic")

    model.train(X_train, y_train)
    model.evaluate(X_val, y_val)
    model.save()

  

if __name__ == "__main__":
    main()
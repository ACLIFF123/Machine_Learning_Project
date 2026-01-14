📰 **Natural Language News Classifier with RAG & Chatbot Interface**

**Project Overview**

This project is a machine learning–powered natural language classifier that categorizes news article titles into predefined topics and explains the reasoning behind each classification. It combines a traditional text classifier with a pre-trained language model, enhanced by Retrieval-Augmented Generation (RAG) for contextual explanations.

The system is designed to mirror real-world ML workflows, including data ingestion, model training, API-ready architecture, and a chatbot-style interface.

**Objectives**

- Build an end-to-end machine learning application

- Use a pre-trained language model alongside a custom classifier

- Train a model on a real-world news dataset

- Explain why a piece of text belongs to a specific category

- Implement RAG using local document retrieval

- Structure the project for production readiness (AWS Lambda + RDS compatible)

**Dataset**

_AG News Topic Classification Dataset_

  This project uses the AG News Topic Classification Dataset, derived from the AG Corpus of News Articles.

_Key Facts:_

- Over 1 million news articles

- Collected from 2,000+ news sources

- Curated for academic research in:

- Text classification

- Information retrieval

- Data mining

The classification benchmark was introduced in:

Xiang Zhang, Junbo Zhao, Yann LeCun (2015)
Character-level Convolutional Networks for Text Classification
NeurIPS 2015

**Categories**

Each news title belongs to one of the following categories:

- World

- Sports

- Business

- Sci/Tech

**What This Project Does**

1. Data Ingestion

- Loads and validates news titles and labels

- Stores data locally (designed to scale to AWS Lambda + RDS)

- Ensures reproducibility and reliability

2. Model Training

- Trains a supervised text classifier on news titles

- Uses vectorization or embeddings derived from a pre-trained language model

- Optimized for fast inference and explainability

3. News Classification

- Given a news title, the model:

- Predicts the most likely category

- Outputs confidence scores (optional)

- Passes the prediction to the language model for explanation

_Example Input:_

"Apple unveils new AI-powered MacBooks"


_Output:_

Category: Sci/Tech
Explanation:
This headline discusses Apple releasing new hardware with artificial intelligence features, which places it within science and technology news.

**_Explanation Engine (Reasoning)_**

_The reasoning layer uses:_

- The classifier’s prediction

- Retrieved examples from similar articles (RAG)

- A pre-trained language model

- This allows the system to justify predictions in natural language, making the model more interpretable and user-friendly.

- Retrieval-Augmented Generation (RAG)

_Uses local document files containing:_

- Sample headlines

- Category descriptions

- Historical examples

- Retrieves relevant context before generating explanations

- Improves accuracy and reduces hallucinations

- Chatbot Interface

- The project includes a chatbot-style interface where users can:

- Enter a news headline

- Receive a category prediction

- Read a human-like explanation of the decision

- This simulates how an ML model might be exposed in a customer-facing application.

**Use Cases**

- News aggregation platforms

- Content moderation systems

- Recommendation engines

- NLP portfolio projects

- Academic research demonstrations

_Disclaimer_

This project uses the AG News dataset strictly for research and non-commercial purposes, in accordance with the dataset’s license.

**Future Improvements**

- Infrastructure as Code: use Terraform to provision AWS resources and deploy an AWS Lambda version of the ingestion/inference pipeline (with environment variables, IAM roles, and optional RDS connectivity)

 - Add CI/CD (GitHub Actions) to automate testing, model packaging, and Lambda deployments

- Fine-tune transformer-based classifiers

- Add multilingual support

- Deploy via REST or GraphQL API

- Integrate vector databases (e.g., FAISS)

- Add confidence calibration and evaluation dashboards

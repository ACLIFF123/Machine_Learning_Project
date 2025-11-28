Machine Learning Project - Natural Language Classifier

Objectives:
    Build a machine learning project which combines a simple classifier with a language model
    Understand how to use a pre-trained language model in a machine learning project
    Create a professional, production-ready machine learning application
    
Overview
Summary:
    Choose a real-world dataset from a curated list
    Ingest and store that data reliably, using a local script which will form the basis for an AWS Lambda with RDS
    Train a model on that data to understand or generate insights
    Create a chatbot interface that uses your trained model
    Add RAG (Retrieval-Augmented Generation) functionality using simple local files

ORIGIN

AG is a collection of more than 1 million news articles. News articles have been gathered from more than 2000 news sources by ComeToMyHead in more than 1 year of activity. ComeToMyHead is an academic news search engine which has been running since July, 2004. The dataset is provided by the academic comunity for research purposes in data mining (clustering, classification, etc), information retrieval (ranking, search, etc), xml, data compression, data streaming, and any other non-commercial activity. For more information, please refer to the link http://www.di.unipi.it/~gulli/AG_corpus_of_news_articles.html .

The AG's news topic classification dataset is constructed by Xiang Zhang (xiang.zhang@nyu.edu) from the dataset above. It is used as a text classification benchmark in the following paper: Xiang Zhang, Junbo Zhao, Yann LeCun. Character-level Convolutional Networks for Text Classification. Advances in Neural Information Processing Systems 28 (NIPS 2015).
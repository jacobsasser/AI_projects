## Clinical Trial Discovery Chatbot

This project builds an AI-powered assistant that helps patients find clinical trials using natural language instead of navigating complex medical databases. It automatically extracts and simplifies trial eligibility criteria, compares them to user-provided information, and returns plain-language explanations powered by a RAG pipeline using BioBERT embeddings and a Pinecone vector database. The system is designed with HIPAA-aligned principles, ensuring no personal health information is stored and all access is securely controlled.

## Hate Speech Classification (TF-IDF vs. BERT)

This project compares a classical TF-IDF + Logistic Regression baseline with a fine-tuned BERT model for binary hate-speech classification. Preprocessing includes token cleaning, stopword removal, and stemming/lemmatization to build sparse TF-IDF features, while BERT uses contextual embeddings and a custom classifier head. Results show that TF-IDF performs well on smaller datasets, but BERT becomes superior as data volume increases, achieving higher accuracy and richer token-level interpretability via SHAP.

## S&P 500 Technology Stock Forecasting (LSTM vs. TCN)

This project forecasts next-day adjusted closing prices for technology-sector stocks using deep learning models trained on merged market + macroeconomic data. After generating 30-day sequences for each stock, both an LSTM and a Temporal Convolutional Network (with stock embeddings) were trained and tuned extensively. Hyperparameter searches and performance visualization reveal that the TCN outperforms the LSTM on RMSE, MAPE, and stability, producing smoother and more accurate predictions across stocks.

## Road Accident Risk (Kaggle Playground)

This project predicts accident risk using tabular deep learning, where categorical features are embedded and combined with normalized numerical variables. A custom PyTorch tabular MLP and Optuna hyperparameter tuning pipeline were developed to optimize dropout, hidden layers, batch size, and learning rate. After full-data training, the model achieved strong leaderboard performance and produced consistent, well-calibrated risk scores for unseen test samples.

## TEDS-D Substance Abuse Treatment Completion Prediction

This project analyzes treatment episode data from the TEDS-D dataset to identify factors associated with completing or dropping out of substance-abuse treatment programs in the Northeastern U.S. region. A full preprocessing and encoding pipeline was built, followed by the training of Logistic Regression, Random Forest, and HistGradientBoosting models with grid-search tuning. The ensemble models achieved the strongest accuracy, and permutation/feature-importance analysis revealed the most influential demographic and clinical predictors of relapse risk.

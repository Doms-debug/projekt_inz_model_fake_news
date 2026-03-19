Fake News Detector 🕵️‍♂️📰
A hybrid system for detecting disinformation in the digital space, combining the speed of statistical classifiers with the deep contextual understanding of Large Language Models (LLMs).

This project was developed as part of the engineering thesis titled "Analiza i detekcja dezinformacji w treściach cyfrowych przy użyciu metod przetwarzania języka naturalnego (NLP) i uczenia maszynowego (ML)" (Analysis and Detection of Disinformation in Digital Content Using Natural Language Processing and Machine Learning Methods) at Vistula University.

📖 About the Project:
This project solves this problem through two integrated stages:Fast Statistical Evaluation: A local classification model evaluates the style and linguistic features of the text, providing a mathematical probability of falsehood.Substantive Verification (RAG): The system searches the internet in real-time to provide the generative model with external context. This allows for objective fact-checking and the correction of classifier errors, such as False Positives.

✨ Key Features:
- ML Classifier (DistilBERT): Fine-tuned on a combined dataset of ISOT and LIAR (over 44,000 records diversified in terms of length and style)
- Fact-Checking Module (RAG): Utilizes the Retrieval-Augmented Generation technique
- The DuckDuckGo search engine retrieves the latest information from the web, which is then synthesized by the expert Llama-3 model
- Interactive UI: An easy-to-use web application built with the Streamlit framework
- Performance Optimization: Utilizes LPU processors via the Groq platform for the LLM and caching mechanisms in Streamlit (Singleton pattern) to ensure near real-time performance

🛠️ Stack:
- Python 3.9+
- PyTorch, Hugging Face Transformers (distilbert-base-uncased model)
- LLM & API: Llama-3-70B-versatile via Groq API
- Retrieval: DuckDuckGo Search API (ddgs library)
- Frontend: Streamlit

📊 Model Performance:
The fine-tuned DistilBERT model achieved excellent results on the test set:
Accuracy: 92.37% 
Recall: 96.00% (high recall minimizes the risk of letting dangerous disinformation slip through) 
F1-Score: 0.9327 

🚀 Local Setup:
Prerequisites:
- Python 3.9+ installed
- An active account and a free API key from Groq

🌐 Live Version
The application has also been deployed and made publicly available using the Hugging Face Spaces infrastructure.
👉 Check out the live app here: https://huggingface.co/spaces/dczajka1/Detektor-Dezinformacji-Inz-App

👨‍💻 Author
Dominik Bartosz Czajka 
Engineering thesis

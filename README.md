# Support Ticket Classification – Task 2

This project focuses on building a machine learning system that automatically classifies IT service support tickets into relevant issue categories using Natural Language Processing (NLP).

The goal is to help support teams route tickets faster and improve response efficiency.

## Dataset
- Dataset Name: IT Service Ticket Dataset
- Text Column: Document
- Target Column: Topic_group
- Data Type: Real-world IT support tickets

## Tools & Technologies
- Python
- Pandas
- Scikit-learn
- TF-IDF Vectorization
- Support Vector Machine (Linear SVM)

## Approach
- Preprocessed ticket text using TF-IDF
- Used unigrams and bigrams for better context understanding
- Trained a Linear SVM classifier for multi-class classification
- Evaluated the model using accuracy and classification report

## Model Performance
- Accuracy: ~85%
- The model performs well across multiple ticket categories such as Access, Hardware, HR Support, and more

## Features
- Automated ticket categorization
- Handles real IT service ticket data
- Supports prediction for new unseen tickets
- Scalable NLP pipeline using sklearn Pipeline

## How to Run
```bash
pip install pandas scikit-learn
python task2_ticket_classification.py

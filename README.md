# TextClassification-using-NLP

This text classification task required classifying biased text data into 17 categories and beat accuracy of baseline Logistic Regression Model as part of the Algorithms for Data Guided Business Intelligence capstone project.  

Features were extracted using multiple methods like tf-idfvectorizer and doc2vec.

**baseline.py** shows the basline model implementation using Logistic Regression.  
**scripts** contains code for the four attempts:  
* **RandomForest.py** : uses tfidf-vectorizer coupled with Random Forest classifier.
* **doc2vec.py** : uses doc2vec coupled with multiple classfier models.
* **first4.py** : checks accuracy and feasibility of classifying data into only top 4 categories.
* **final.py** : uses learnings from above attempts, coupled with manually engineered features and cross validation to beat baseline model by 11%.

**data** consists of training and testing set.




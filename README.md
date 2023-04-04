# Drug_reviews_condition_classification-main.
About Data : 
         no of data is 215063.
         The data is split into a 75% train and 25% test partition and stored in two tsv files.
         
         
I have built a engine for classification of patients condition using drugs review.We predict the patient condition whether is suffering from high blood pressure and Depression and any other disease. Then will recommend him a suitable drug.

- Reviews are very important to get the overview of product whether it is service , offering or products.
- Reviews also plays a very important role in healthcare domain especially in terms of drugs.
- By analyzing the reviews , we can get the understanding of the drug effectiveness and its side efects.
- But in this project , we will classify the condition of patient based on his reviews so that we can reccomend him a suitable drug.


Steps of NLP pipeline :-
- tokenize the sentences
- clean reviews :  
-                Remove punctuation
-                Remove special characters
-  Create bag of words model to vectorize.
-  Apply Ml algorithms , Naive Bayes and Passive Aggressive Classifier.
-  Create TFIDF model to vectorize. 
-  Apply Ml algorithms , Naive Bayes and Passive Aggressive Classifier.  
-  compare between Bag of words and TFIDF vetorizer in terms of accuracy.                 

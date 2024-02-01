"""
HW1 - Movie Review Classification
Name - Saksham Nayyar
Username - Sakshamnayyar

"""


import pandas as pd
import re
import string

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier

#Constants
TRAIN_FILE = "train_new.txt"
TEST_FILE = "test_new.txt"
OUTPUT_FILE = "format.dat"
train_sentiments = []
train_reviews = []

def clean_train_dataset(text):
    """
    This funtions cleans the training dataset for preprocessing 
    and extracts the reviews and sentiments to store them in the list
    train_review and train_sentiments(resp.).

    Parameters
    ----------
    text(Object): One row of the dataset. 

    Returns
    -------
    None.

    """
    # convert to string
    combined_string = str(text)
    
    #remove any html tags in the string
    combined_string = re.sub(r'<.*?>', '', combined_string)
    
    #remove '#EOF' from the string
    combined_string = combined_string.replace("#EOF","")
    
    #removing '-' from whole string leaving the starting part that includes the sentiment
    combined_string = re.sub(r'-(?![0-9])',' ', combined_string)
    
    #replacing \t with a comma
    combined_string = combined_string.replace("\t",", ")
    
    #extracting review from the combined string.
    review = re.sub(r'^[+-]?\d+,\s','',combined_string)
    
    #remove any numbers from review
    review = re.sub(r'\d+','',review)
    
    #converting to lower case and storing review in reviews list
    train_reviews.append(review.lower())
    
    #storing sentiment from combined_string in sentiments list.
    train_sentiments.append(re.search(r'[+-]?\d+',combined_string).group())
    

def clean_test_dataset(text):
    """
    This function cleans testing dataset for preprocessing.

    Parameters
    ----------
    text (Object): single row of testing dataset.

    Returns
    -------
    string: cleaned string.

    """
    
    # convert to string
    review = str(text)

    #remove any html tags in the string
    review = re.sub(r'<.*?>', '', review)
    
    #remove '#EOF' from the string
    review = review.replace("#EOF","")
    
    #remove numbers from string
    review = re.sub(r'\d+','',review)
    
    #convert to lowercase and return
    return review.lower()

def preprocess_text(text):
    """
    This API is used to preprocess the training as well as testing dataset.
    it basically removes punctiations, special characters and stopwords.
    After that it applies stemming to the words.

    Parameters
    ----------
    text (String) : single row of the dataset.

    Returns
    -------
    String: String after pre-processing the input.

    """
    # Tokenization i.e.convert document to a single word.
    words = word_tokenize(text)
    
    # Remove punctuation and special charcters
    words = [word for word in words if word.isalnum() and 
             word not in string.punctuation]
    
    # Remove stopwords
    words = [word for word in words if word not in stopwords.words('english')]
    
    # Apply stemming to each word
    words = [PorterStemmer().stem(word) for word in words]
    
    return ' '.join(words)


def main():
    #import train dataset from file "train_new.txt"
    train_data = pd.read_csv(TRAIN_FILE,"    ",header = None, names = ['data'])
    
    #import test dataset from file "test_new.txt"
    test_reviews = pd.read_csv(TEST_FILE,"    ",names = ['data'])

    #cleaning train dataset 
    train_data['data'].apply(clean_train_dataset)
    train_data['reviews'] = train_reviews
    train_data['sentiments'] = train_sentiments
    
    #started cleaning test dataset
    test_reviews['data'] = test_reviews['data'].apply(clean_test_dataset)
    
    #preprocessing train dataset
    train_data['reviews'] = train_data['reviews'].apply(preprocess_text)
    
    #preprocessing test dataset
    test_reviews['data'] = test_reviews['data'].apply(preprocess_text)
    
    # Created a TF-IDF vectorizer object according to the requirement.
    tfidf_vectorizer = TfidfVectorizer(max_df=0.7,ngram_range=(1,2),
                                       min_df=0.004, max_features=1500)

    # Transform train_data using TF-IDF vectorizer to create X_train matrix.
    X_train = tfidf_vectorizer.fit_transform(train_data['reviews'])
    
    # Converted the sentiments to int
    y_train = train_data['sentiments'].astype(int)

    # Transform the test_reviews data using the same TF-IDF vectorizer to create X_test matrix
    X_test = tfidf_vectorizer.transform(test_reviews['data'])

    # Created a KNN classifier
    knn_classifier = KNeighborsClassifier(n_neighbors=150,weights='distance'
                                          ,n_jobs=-1)
    
    # feeded the data to the classifier to make the model.
    knn_classifier.fit(X_train, y_train)
    
    # Make predictions on the test data
    y_pred = knn_classifier.predict(X_test)
    
    # Save the predictions to "format.dat" file. 
    y_pred.tofile(OUTPUT_FILE,'\n')
    
    
if __name__ == "__main__":
    main()
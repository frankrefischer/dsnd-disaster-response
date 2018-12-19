import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
import os
import pandas as pd
import pickle
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
import sqlalchemy
import sys
from time import time

def report_time(func):
    """A decorator function to report the elapsed time for executing the decorated function."""
    
    def wrapper(*args, **kwargs):
        t0 = time()
        x = func(*args, **kwargs)
        dt = time() - t0
        if dt > 1:
            print("elapsed time: {:.3f}s".format(dt))
        else:
            print("elapsed time: {:.3f}ms".format(dt*1000))
        return x
    return wrapper

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        @report_time
        def train_model(X, Y):
            model.fit(X, Y)
        train_model(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')

@report_time
def load_data(database_filepath):
    """Load disaster reponse data from a sqlite database file.
    
    All data is expected to be stored in a table named 'DisasterResponse'.
    
    Keyword arguments:
    database_filepath -- load this sqlite database file
    
    Return Value: X,y
    X              -- the messages as a series
    y              -- a dataframe with only the category columns
    category_names -- all the category names of y 
    """
    assert os.path.isfile(database_filepath), "no such file: {}".format(database_filepath)
    
    sqlite_engine = sqlalchemy.create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_query('SELECT * FROM DisasterResponse', sqlite_engine)
    X = df.message
    y = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    category_names = y.columns.values
    return X, y, category_names

@report_time
def build_model():
    """Creates the model."""
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier())),
    ])

    parameters = {
        'vect__max_df': [0.75],
        'tfidf__use_idf': [True],
        #'clf__max_depth': [3]

        #'vect__max_df': (0.5, 0.75, 1.0),
        #'tfidf__use_idf': (True, False),
        #'clf__max_depth': [3, None]
    }

    return GridSearchCV(pipeline, param_grid=parameters, cv=2, verbose=99)

@report_time
def evaluate_model(model, X_test, Y_test, category_names):
    """Compare the predictions of model for X_test to Y_test.
    
    Prints a classification report for every category.
    
    Keyword arguments:
    model          -- the model to evaluate
    X_test         -- the test set of inputs
    Y_test         -- the true labels
    category_names -- the message category names
    """
    Y_pred = model.predict(X_test)
    
    for i in range(len(Y_test.columns)):
        print("=== {}. {} ===".format(i+1, Y_test.columns[i]))
        print(classification_report(Y_test.iloc[:,i], Y_pred[:,i]))

@report_time
def save_model(model, model_filepath):
    """Saves the model to a pickle file.
    
    Keyword arguments:
    model          -- the model to save
    model_filepath -- save the model to this file
    """
    
    pickle.dump(model, open(model_filepath, 'wb'))

DIGITS = re.compile('[0-9]+')
PUNCTUATION_CHARS = re.compile("[.,;:''()#]")
LEMMATIZER = nltk.stem.WordNetLemmatizer()
ENGLISH_STOPWORDS = nltk.corpus.stopwords.words('english')

def tokenize(text):
    """Tokenize the input text.
    
    Keyword arguments:
    text -- the text to tokenize
    
    Return value:
    list of tokens
    """
    def lemmatize_token(tok):
        return LEMMATIZER.lemmatize(tok).lower().strip()
    def is_not_an_english_stopword(w):
        return w not in ENGLISH_STOPWORDS

    text = DIGITS.sub('numberplaceholder', text)
    text = PUNCTUATION_CHARS.sub(' ', text)
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatize_token(t) for t in tokens if is_not_an_english_stopword(t)]
    
    return tokens

if __name__ == '__main__':
    main()
import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Histogram
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    """Tokenize a text into a list of tokens."""

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterResponse', engine)

# load model
model = joblib.load("../models/DisasterResponse.pickle")


@app.route('/')
@app.route('/index')
def index():
    """Display cool visuals and receive user input text for model."""
    
    graphs = [
        make_graph_distribution_of_message_genres(),
        make_graph_distribution_of_message_genres(),
        make_graph_distribution_of_message_sizes(),
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


@app.route('/go')
def go():
    """Handle user query and display model results."""

    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )

def make_graph_distribution_of_message_genres():
    """Make plotly graphobject fot distribution of message genres."""

    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # create visuals    
    return {
            'data': [Bar(x=genre_names, y=genre_counts)],
            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': { 'title': "Count" },
                'xaxis': { 'title': "Genre" },
            }
    }

def make_graph_distribution_of_message_sizes():
    """Make plotly graphobject fot distribution of message sizes."""

    msgsizes = df.message.str.len()
    return {
        'data': [Histogram(x=msgsizes)],
        'layout': {
            'title': 'Distribution of Message Sizes',
            'yaxis': { 'title': 'Count' },
            'xaxis': { 'title': 'log Size', 'type': 'log' },
        }
    }       
    
def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
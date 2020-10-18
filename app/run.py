import json
import plotly
import pandas as pd
import string
import collections

import nltk 
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
nltk.download('stopwords')

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine



app = Flask(__name__)

def tokenize(text):
    
    no_punct = ""
    punct=string.punctuation
    for char in text:
        if char not in punct:
            no_punct = no_punct + char
    text=no_punct
    stop_words = nltk.corpus.stopwords.words("english") 
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
        clean_tokens=[word for word in clean_tokens if word not in stop_words]

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('messages', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    #Top 5 Catigories
    top_5_cat_cnt = df.iloc[:,4:].sum().sort_values(ascending=False).head(5)
    top_5_cat_nm = list(top_5_cat_cnt.index)

    #Bottom 5 catigories
    bottom_5_cat=df.iloc[:,4:].sum().sort_values(ascending=True).head(5)
    bottom_5_cat_names=genre_names = list(bottom_5_cat.index)

    
    # Analysis of most common word counts
    word_list=[]; 
    message_list=df['message'].tolist()
    for message in message_list:
        t_mess=tokenize(message)
        word_list.extend(t_mess)
    counter = collections.Counter(word_list)
    twenty_most_common_words_dict = dict(counter.most_common(20))   
    twenty_most_common_words_key=list(twenty_most_common_words_dict.keys())
    twenty_most_common_words_value=list(twenty_most_common_words_dict.values())  

   
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
                'data': [
                    Bar(
                        x=top_5_cat_cnt,
                        y=top_5_cat_nm,
                        orientation='h'
                       
                    )
                ],

                'layout': {
                    'title': 'Top Categories',
                    'yaxis': {
                        #'title': ""
                    },
                    'xaxis': {
                        'title': "Count"
                    }
                }
            },

        {
            'data': [
                Bar(
                    x=bottom_5_cat,
                    y=bottom_5_cat_names,
                    orientation='h'
                )
            ],

            'layout': {
                'title': 'Bottom Categories',
                'yaxis': {
                    'title': ""
                },
                'xaxis': {
                    'title': "Count"
                }
            }
        },
      {
            'data': [
                Bar(
                    x=twenty_most_common_words_key,
                    y=twenty_most_common_words_value
                )
            ],

            'layout': {
                'title': "Distribution of Top 20 Word Counts",
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Word"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
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


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
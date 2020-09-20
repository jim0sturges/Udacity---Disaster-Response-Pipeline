import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=DeprecationWarning)
    import sys
    import pandas as pd
    import numpy as np
    from IPython.display import display
    import string                          #library of string constants
    import re
    from sqlalchemy import create_engine
    import nltk                            # natural language tool kit
    nltk.download('stopwords')
    import pickle

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer, precision_recall_fscore_support
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import TfidfTransformer,CountVectorizer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.multioutput import MultiOutputClassifier
    from sklearn.model_selection import GridSearchCV
    from sklearn.cross_validation import train_test_split


def load_data(database_filepath):
    """
    Description: This function creates a sqlalchemy database engine
        and reads in the cleaned data from the messages table previously stored.
    Arguments: the database path
    Returns: the message series (X), the Dataframe of features and feature df column names.
    """

    engine = create_engine('sqlite:///DisasterResponse.db')
    df = pd.read_sql_table('messages', engine)
    X = df['message']
    Y = df.drop(['message', 'genre', 'id', 'original'], axis=1)
    cols=list(Y.columns)
    return(X,Y,cols)
    


def tokenize(text):
    """
    Description: This function reads a text string
                    removes punctuation
                    converts to lower case
                    strips out white space
                    tokenizes the text into a word list
                    Stems the word list
                    removes stop words

    Arguments: Initial test string
    Returns: clean list of word tokens
    """

    stop_words = nltk.corpus.stopwords.words("english")

    # Normalize text
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower().strip())

    #tokenize
    words = nltk.word_tokenize (text)
    clean_tokens = [nltk.PorterStemmer().stem(w) for w in words if w not in stop_words]

    return clean_tokens


def report_results(y_test, y_pred):
        
     """
     Description: This function reads in the Y test and the Y predicted values
                  it used the sklearn precision_recall_fscore_support function
                  to compute the fscore, precision and recall values for each catigory
                  it places them in a dataframe for nice formating
                  lastly it compute the average of these score to provide and view
     Arguments: the y test and y predicted values
     Returns: a summary report
     """
     report = pd.DataFrame(columns=['Category', 'f_score', 'precision', 'recall'])
     indx=0
     for cat in y_test.columns:
        precision, recall, f_score, support = precision_recall_fscore_support(y_test[cat],y_pred[:,indx], average='weighted')
        report.at[indx,'Category']=cat
        report.at[indx,'f_score']=f_score
        report.at[indx,'precision']=precision
        report.at[indx,'recall']=recall
        indx=indx+1
     report.set_index('Category', inplace=True)
     display(report)
    
     rd={}
     rd['Precision (averaged)']=report['precision'].mean()
     rd['Recall (averaged)']=report['recall'].mean()
     rd['F_score (averaged)']=report['f_score'].mean()
     display(pd.DataFrame.from_dict(rd,orient='index').rename(columns={0:"Score"}))
    
     return  
    
    

def build_model():
    """
     Description: This function builds a pipeline model using Sklearns Pipeline function and the RandomForest Classifer
        It then improves the model by runing a gridsrch to find the best parameters
        ref: https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html
        ref: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
        ref: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
     Arguments: none
     Returns: pipeline model
     """
    forest_clf = RandomForestClassifier(n_estimators=10)
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer = tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(forest_clf))
        ])
    
    # improve model with Grdisrch
    parameters ={'vect__min_df': [5],
                 'tfidf__use_idf':[True],
                 'clf__estimator__n_estimators':[5], 
                 'clf__estimator__min_samples_split':[2, 5]}

    pipeline = GridSearchCV(pipeline, param_grid = parameters,  verbose = 1)

    return(pipeline)
    
    
def evaluate_model(pipeline, X_test, Y_test, category_names):
    """
    Description: This function calls a reporting function that reports (prints) statistics
        regarding on effectiveness of the pipeline model provided
    Argements: The pipeline model
                The X_Test seris, the Y_test series and the catigory names
    Returns: none
    """
  
    y_pred = pipeline.predict(X_test)
    report_results(Y_test, y_pred)
    return

    
    
def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))
    return


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
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


if __name__ == '__main__':              
    main()
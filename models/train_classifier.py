# train_classifier.py

###### 1. import libraries and load data frome database ######

## import libraries
import pandas as pd
import numpy as np
from IPython.display import display
import string                          #library of string constants
import re
from sqlalchemy import create_engine
import nltk                            # natural language tool kit
from sklearn.model_selection import GridSearchCV
import pickle                            

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

## load data from database
## next two lines for local environment to be commented out on submission
#os.path.abspath(os.getcwd())
#os.chdir('C:\\Users\\Jim.000\\Documents\\Udacity Data Scientist Class\\Pipeline\\sturges_disaster_response_pipeline_project/')

engine = create_engine('sqlite:///data/DisasterResponse.db')
df = pd.read_sql_table('messages', engine)
X = df['message']
Y = df.drop(['message', 'genre', 'id', 'original'], axis=1)
cols=list(Y.columns)
print('DisasterResponse.db read in')
print()

###### 2. Write a tokenization function to process your text data ######

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
    #stem
    #A processing interface for removing morphological affixes from words. This process is known as stemming.
    # (https://www.nltk.org/api/nltk.stem.html)
    clean_tokens = [nltk.PorterStemmer().stem(w) for w in words if w not in stop_words]
    return clean_tokens

###### 3. Build a machine learning pipeline ######   

print('3. Building machine learning pipeline')
print()

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer,CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
forest_clf = RandomForestClassifier(n_estimators=10)
pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer = tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(forest_clf))
])

print('     machine learning pipeline built')
print()

###### 4. Train pipeline ######

print('4: begin training pipeline')
print()

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.33, random_state = 42)

pipeline.fit(X_train.values, Y_train.values)

#pipeline.fit(X_train.as_matrix(), Y_train.as_matrix())
y_pred = pipeline.predict(X_test)


###### 5. Test your model ######

print('5: testing model')
print()


def report_results(y_test, y_pred):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer
    from sklearn.metrics import precision_recall_fscore_support
    
    #display(y_pred,Y_test)
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
    for indx, cat in enumerate(y_test.columns):
        idx=indx+1
        precision, recall, f_score, support = precision_recall_fscore_support(y_test[cat], y_pred[:,indx], average='weighted')
        report.at[idx,'Category']=cat
        report.at[idx,'f_score']=f_score
        report.at[idx,'precision']=precision
        report.at[idx,'recall']=recall

    rd={}
    rd['Precision (averaged)']=report['precision'].mean()
    rd['Recall (averaged)']=report['recall'].mean()
    rd['F_score (averaged)']=report['f_score'].mean()
    display(pd.DataFrame.from_dict(rd,orient='index').rename(columns={0:"Score"}))
    report.set_index('Category', inplace=True)
    return report

print('intial model results')
print(report_results(Y_test, y_pred))

###### 6. Improve your model ######

def improve_model(pipeline,X_train,Y_train):


    """
    Description: This function reads in a previously built pipeline from the sklearn library
    (https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)
    then a gridsearch (https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)
    using diferent parameter (currently hardcoded in this model but they could be passed in)
    
    Inputs:
    pipeline: predefined using sklearn
    X: iterable Training data
    y: iterable Training data

    Arguments: the y test and y predicted values
    Returns: fitted, tunned model
    """
   
    parameters = {'vect__min_df': [5],
                  'tfidf__use_idf':[True, False],
                  'clf__estimator__n_estimators':[25], 
                  'clf__estimator__min_samples_split':[2, 5]}

    print()
    print()
    print('6. Improving model with Gridsrch')
    print()
    cv = GridSearchCV(pipeline, param_grid = parameters,  verbose = 1)

    # Find best parameters
    np.random.seed(81)
    tuned_model = cv.fit(X_train, Y_train)
    tuned_model=tuned_model.best_estimator_
    return tuned_model

tuned_model=improve_model(pipeline,X_train,Y_train)

###### 7. Test your model ######

y_pred = tuned_model.predict(X_test)

print("     improved model results")
print(report_results(Y_test, y_pred))
print()
###### 8. Try improving your model further. Here are a few ideas: ######
## no improvement so skipped this step since not required in the rubric

###### 9. Export your model as a pickle file ######

pickle.dump(tuned_model,open('models/classifier.pkl','wb'))

print('8. model saved to models/classifier.pkl')
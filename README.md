# Disaster Response Pipeline Project

##### Project by: James Sturges
This is the second project of the Udacity Data scientist nanodegree program

### Purpose:
This project demonstrates the creation of  2 pipelines which classifies a set of disaster recovery messages into categories using machine learning techniques. The classification model once developed and tuned is stored as a pickle file. A web app provide a brief explortion of the model and provide the user a way to input for classification.
The steps include:
1. An extraction, transformation, and loading (ETL) pipline to clean and load the messaginging data
2. A machine learning pipeline using sklearn techniques to create and improve a meachine learning model relative to the messaging data.
3. Lastly, we provide visualizations of the analysis. Which can be viewed in a local environment at local host://3001.


### Instructions:

#### Libraries
Running this code will require python 3.0 and the following libraries:
pandas, numpy, from sqlalchemy import create_engine, os,  sys, string , re, nltk,
, pickle, iPython
from sklearn: 
* from model_selection import GridSearchCV
* from pipeline import Pipeline
* from feature_extraction.text import TfidfTransformer,CountVectorizer
* from ensemble import RandomForestClassifier
* from multioutput import MultiOutputClassifier 
* from model_selection import train_test_split
* from metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer
* from metrics import precision_recall_fscore_support

##### Program Modules

###### Process_data.py
This module:
1. reads in data from csv files for both the messages and catigories.
2. It cleans the data
3. Stores the cleaned data to an SQL lite database.

###### train_classifier.py
This module:
1. Imports the SQL lite db (created above) into a pandas dataframe
2. It uses sklearn models to create and tune a machine learning model for catogorizing messages
3. The resulting fitted model is written out a a pickle file.

###### iPython Notebooks
This notebooks were used in the deveopment of the process and classifier modules above and are provided purely for docmentation purposes.

###### app/templates
these html templates are used by the web app


#### Data
The project provides two data files:
1.  messages.csv - This files contains the disaster recovery messages, the original mess and a genre
2.  categories.csv - this file provides the catigories and associated catigory numbers

#### Running the program

1. Navigate to the to level of the project directory (contains read me file)

2. Run process and train modules
*  To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
* To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

3. Run the web App
    * cd to the app directory ("cd/app") (no quotes)
    * type python run.py
    * go to your browse and type in the search window "http://localhost::3001"
    
    note: I could not get firefox to work with this, I had to use chrome
    


#### If you are running this in Udacity's IDE workspace 
###### (the following instructions are provided by UDACITY, I was unable to get the IDE workspace to function consistently, dispite the effort of the technical mentors. Consequently, I cannot vouch for the below working reliably.


* open a new termination window by clicking on the plus (+) sign
* Type in "env|grep WORK" (without quotes)
* you should be given back:
  + WORKSPACEDOMAIN=udacity-student-workspaces.com
  + WORKSPACEID=viewxxxxxxxxx , where then x's have an alphanumeric value
* Now create a new URL value: "https://viewxxxxxxxxx-3001.udacity-student-workspaces.com" (no quotes) replacing the x's with the new view value
* paste that url into your browser to get to the visual rendition of your code


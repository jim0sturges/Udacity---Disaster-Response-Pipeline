# Disaster Response Pipeline Project

##### Project by: James Sturges
This is the second project of the Udacity Data scientist nanodegree program
In this project we create 2 pipelines. 
1. An extraction, transformation, and loading (ETL) pipline to clean and load our messaginging data
2. A machine learning pipeline using sklearn techniques to create and improve a meachine learning model relative to the messaging data.

Lastly, we provide visualizations of the analysis. Which can be viewed in Udacity Student workspaces.


### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
type 'cd app' (no quotes) to change to the app directory
then type: `python run.py` (no quotes)

3. Go to hrunttp://0.0.0.0:3001/

## If you are running this in Udacity's IDE workspace

* open a new termination window by clicking on the plus (+) sign
* Type in "env|grep WORK" (without quotes)
* you should be given back:
  + WORKSPACEDOMAIN=udacity-student-workspaces.com
  + WORKSPACEID=viewxxxxxxxxx , where then x's have an alphanumeric value
* Now create a new URL value: "https://viewxxxxxxxxx-3001.udacity-student-workspaces.com" (no quotes) replacing the x's with the new view value
* paste that url into your browser to get to the visual rendition of your code


https://view6914b2f4-3001.udacity-student-workspaces.com



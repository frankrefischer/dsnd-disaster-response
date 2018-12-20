# Disaster Response Pipeline Project

During a disaster the helping personnel receives lots of online text messages, but they have no time
to analyse if they are relevant.

With this web app you may enter a text message and get indicated, what kind of message it is.

The classication model was trained on a dataset of more than 26000 real world text messages sent in a disaster context.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
	Or if you are working in a udacity workspace environment: run `app/app-url.sh`
    and click on the displayed url


import pandas as pd
import sqlite3
import string
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import GridSearchCV
import joblib
import tweepy
import re
from string import punctuation


class TwitterSentiment:

    def __init__(self, **kwargs):
        self.consumer_key = kwargs.get('consumer_key', '')
        self.consumer_secret = kwargs.get('consumer_secret', '')
        self.access_token = kwargs.get('access_token', '')
        self.access_token_secret = kwargs.get('access_token_secret', '')

        self.auth = tweepy.OAuthHandler(self.consumer_key, self.consumer_secret)
        self.auth.set_access_token(self.access_token, self.access_token_secret)

        self.api = tweepy.API(self.auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)
        try:
            self.api.verify_credentials()
            print("Authentication OK")
        except tweepy.error.TweepError as ex:
            print("Error during authentication")

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        self.conn = sqlite3.connect('./data/data.db')
        cursor = self.conn.cursor()
        cursor.execute("DROP TABLE IF EXISTS tweets")
        self.conn.commit()
        cursor.execute("CREATE TABLE tweets(date none, location none, followers none, friends none, message none)")
        self.conn.commit()
        cursor.close()

    def train(self, train_dataset_path: string = './data/training/training_data.csv'):
        data = pd.read_csv(train_dataset_path, error_bad_lines=False)
        data.columns = ['label', 'id', 'date', 'source', 'user', 'text']
        data = data.drop(['id', 'source', 'date', 'user'], axis=1)

        positives = data['label'][data.label == 4]
        neutrals = data['label'][data.label == 2]
        negatives = data['label'][data.label == 0]

        print('Number of positive tagged sentences is:  {}'.format(len(positives)))
        print('Number of neutral tagged sentences is:  {}'.format(len(neutrals)))
        print('Number of negative tagged sentences is: {}'.format(len(negatives)))
        print('Total length of the data is:            {}'.format(data.shape[0]))

        print("\nTraining...")

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(data['text'], data['label'], test_size=0.2)

        pipeline = Pipeline([
            ('bow', CountVectorizer(strip_accents='ascii', stop_words='english', lowercase=True)),
            ('tfidf', TfidfTransformer()),
            ('classifier', MultinomialNB()),
        ])

        parameters = {'bow__ngram_range': [(1, 1), (1, 2)],
                      'tfidf__use_idf': (True, False),
                      'classifier__alpha': (1e-2, 1e-3),
                      }

        grid = GridSearchCV(pipeline, cv=10, param_grid=parameters, verbose=1)
        grid.fit(self.X_train, self.y_train)

        print("\nBest Model: %f using %s" % (grid.best_score_, grid.best_params_))
        print('\n')
        means = grid.cv_results_['mean_test_score']
        stds = grid.cv_results_['std_test_score']
        params = grid.cv_results_['params']
        for mean, stdev, param in zip(means, stds, params):
            print("Mean: %f Stdev:(%f) with: %r" % (mean, stdev, param))

        print("\nTraining finished!")
        print("\n")

        joblib.dump(grid, "./data/twitter_sentiment.pkl")

    def clean_message(self, tweet):
        tweet = re.sub(r'\&\w*;', '', tweet)
        tweet = re.sub('@[^\s]+', '', tweet)
        tweet = re.sub(r'\$\w*', '', tweet)
        tweet = tweet.lower()
        tweet = re.sub(r'https?:\/\/.*\/\w*', '', tweet)
        tweet = re.sub(r'#\w*', '', tweet)
        tweet = re.sub(r'[' + punctuation.replace('@', '') + ']+', ' ', tweet)
        tweet = re.sub(r'\b\w{1,2}\b', '', tweet)
        tweet = re.sub(r'\s\s+', ' ', tweet)
        tweet = tweet.lstrip(' ')
        tweet = ''.join(c for c in tweet if c <= '\uFFFF')
        return tweet

    def search_tweets(self, query, item_limit: int = 100):
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM tweets")
        self.conn.commit()
        tweets = tweepy.Cursor(self.api.search, q=query, lang="en", tweet_mode='extended').items(item_limit)
        for tweet in tweets:
            try:
                fields = [tweet.created_at, tweet.user.location, tweet.user.followers_count, tweet.user.friends_count,
                          tweet.full_text]
                cursor.execute("INSERT INTO tweets VALUES (?,?,?,?,?)", fields)
            except tweepy.error.TweepError as ex:
                pass
        cursor.close()

        df_twtr = pd.read_sql_query("SELECT * FROM tweets", self.conn)

        df_twtr['date'] = pd.to_datetime(df_twtr['date'])
        df_twtr = df_twtr.sort_values(by='date', ascending=True)
        df_twtr = df_twtr.reset_index().drop('index', axis=1)
        df_twtr.head()

        df_twtr['message'] = df_twtr['message'].apply(self.clean_message)

        return df_twtr

    def prediction(self, query, item_limit: int = 100):
        df_twtr = self.search_tweets(query, item_limit)
        model_NB = joblib.load("./data/twitter_sentiment.pkl")
        y_predictions = model_NB.predict(self.X_test)
        print('\n')
        print('Accuracy score: ', accuracy_score(self.y_test, y_predictions))
        print('Confusion matrix: \n', confusion_matrix(self.y_test, y_predictions))
        print('\n')
        print('0 = negative, 2 = neutral, 4 = positive')
        print(classification_report(self.y_test, y_predictions))

        tweet_preds = model_NB.predict(df_twtr['message'])
        df_tweet_predictions = df_twtr.copy()
        df_tweet_predictions['predictions'] = tweet_preds

        neg = df_tweet_predictions.predictions.value_counts()[0]
        neu = df_tweet_predictions.predictions.value_counts()[2]
        pos = df_tweet_predictions.predictions.value_counts()[4]

        print('Model predictions: Positives - {}, Neutrals - {}, Negatives - {}'.format(pos, neu, neg))
        df_tweet_predictions.to_pickle('./data/tweet_predicts_df.p')

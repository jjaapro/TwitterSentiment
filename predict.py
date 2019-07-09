from lib.TwitterSentiment import TwitterSentiment


kwargs = {'consumer_key': 'xxxxxxxxxxxxxxxxxxxx',
          'consumer_secret': 'xxxxxxxxxxxxxxxxxxxx',
          'access_token': 'xxxxxxxxxxxxxxxxxxxx',
          'access_token_secret': 'xxxxxxxxxxxxxxxxxxxx'}

twitter_sentiment = TwitterSentiment(**kwargs)

twitter_sentiment.train()

search = input("Search:")
if search != "":
    twitter_sentiment.prediction(search)

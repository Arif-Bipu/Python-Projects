#Arif's use of Twitter Sentiment Analysis Challenge for Learn Python for Data Science #2 by @Sirajology on Youtube
import tweepy
from textblob import TextBlob
import csv

consumer_key = ''
consuper_secret = ''

access_token = ''
access_token_secret = ''

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

public_tweets = api.search('biden')


with open('twitter.csv', 'w') as csvfile:
	filewriter = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
	filewriter.writerow(['Tweet', 'Sentiment'])

	for tweet in public_tweets:
		if(TextBlob(tweet.text).sentiment.polarity < 0):
			pol='Negative'

		else:
			pol='Positive'

		filewriter.writerow([tweet.text.encode('utf-8'), pol])
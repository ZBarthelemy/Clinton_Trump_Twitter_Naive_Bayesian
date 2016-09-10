from twython import Twython
import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import twitter_samples
import pickle
import sys

def get_words_in_tweets(tweets):
	all_words = []
	for (words, sentiment) in tweets:
		all_words.extend(words)
	return all_words

def get_word_features(wordlist):
	wordlist = nltk.FreqDist(wordlist)
	word_features = wordlist.keys()
	return word_features

def extract_features(document):
	document_words = set(document)
	features = {}
	for word in word_features:
		features['contains(%s)' % word] = (word in document_words)
	return features

if len(sys.argv) > 1:
	CONSUMER_KEY = 'JKpOjndKU2xEmYZpIsDrBCjYp'
	CONSUMER_SECRET = 'oisVlCx12XxAX0EZa07x9srSdZP7jRBYj1PGs77iZs78tIxe46'

	f = open(sys.argv[1], 'rb')
	classifier = pickle.load(f)
	f.close()

	twitter = Twython(CONSUMER_KEY, CONSUMER_SECRET)
	count = 0

	geocodes = ['48.898922,-72.736576,15mi',
				'40.743036,-73.297565,6mi',
				'40.830389,-73.507667,4mi',
				'40.697914,-73.606245,4mi',
				'40.720184,-73.794984,2mi',
				'40.762925,-73.829531,1mi',
				'40.6631,-73.9095,4mi',
				'40.725435,-73.995845,1mi',
				'40.769351,-73.988281,0.9mi',
				'40.788507,-73.977156,0.9mi',
				'40.5795,-74.1502,4mi',
				'40.79667,-73.92194,1.7mi',
				'40.8417,-73.9394,2.3mi',
				'40.8497,-73.8331,2mi',
				'40.825,-73.9059,2mi',
				'41.0762,-73.8587,5mi',
				'41.1507,-73.9454,6mi',
				'41.0051,-73.7846,5mi',
				'41.4020,-74.3243,10mi',
				'41.4389329,-73.8075079,8mi',
				'41.4351,-73.7949,6mi',
				'41.9270,-73.9974,35mi',
				'42.7289,-73.8137,20mi',
				'44.2239,-74.4641,80mi',
				'43.1009,-75.2327,20mi',
				'42.0987,-75.9180,15mi',
				'42.0970,-79.2353,5mi',
				'42.0836,-78.4299,5mi',
				'42.0898,-76.8077,5mi',
				'42.4440,-76.5019,5mi',
				'43.0481,-76.1474,20mi',
				'43.1610,-77.6109,25mi',
				'42.8864,-78.8784,15mi',
				'42.5565,-78.1528,30mi']

	print "geocode\tlocation\tpos\tneg"
	for geocode in geocodes:
		tweets = []
		locs = []
		max_id = None
		count = -1
		ids = []
		while len(tweets) < 1000 and count != 0:
			count = 0
			for status in twitter.search(q='trump', count = 1000, geocode=geocode, max_id=max_id)["statuses"]:
				user = status["user"]["screen_name"].encode('utf-8')
				text = status["text"].encode('utf-8')
				loc = status["user"]["location"].encode('utf-8')
				if loc != "":
					locs.append(loc)
				words_filtered = [e.lower() for e in text.split() if len(e) >= 3]
				if status["id"] in ids:
					print "duplicate!!!"
				tweets.append((words_filtered, text))
				ids.append(status["id"])
				max_id = status["id"]-1
				count += 1
		word_features = get_word_features(get_words_in_tweets(tweets))
		posCount = 0
		negCount = 0
		for (words, tweet) in tweets:
			sentiment = classifier.classify(extract_features(words))
			if sentiment == 'pos':
				posCount += 1
			elif sentiment == 'neg':
				negCount += 1
		if len(locs) != 0:
			print geocode,"\t",max(set(locs), key=locs.count),"\t",posCount,"\t",negCount
		else:
			print geocode,"\t","no name","\t",posCount,"\t",negCount
else:
	strings = twitter_samples.strings('positive_tweets.json')
	strings2= twitter_samples.strings('negative_tweets.json')

	pos = []
	for word in strings:
		pos.append((word, "pos"))

	neg = []
	for word in strings2:
		neg.append((word, "neg"))

	tweets = []

	for (words, sentiment) in pos + neg:
		words_filtered = [e.lower() for e in words.split() if len(e) >= 3]
		tweets.append((words_filtered, sentiment))

	word_features = get_word_features(get_words_in_tweets(tweets))

	training_set = nltk.classify.apply_features(extract_features, tweets)
	print "Training on",len(training_set),"tweets"
	classifier = NaiveBayesClassifier.train(training_set)

	f = open('my_classifier.pickle', 'wb')
	pickle.dump(classifier, f)
	f.close()
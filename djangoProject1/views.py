from django.shortcuts import render;
import joblib
import nltk
import re
import tweepy


def home(request):
    return render(request, "home.html")

def predict1(request):
    classifier = joblib.load('svmClassifier.pkl')
    tweet = request.GET['tweets']
    res = pred(tweet, classifier)
    return render(request, "predict1.html", {"results":res})

def predict2(request):
    classifier = joblib.load('svmClassifier.pkl')
    user = request.GET['username']
    count = 100
    pcount = 0
    ncount = 0

    consumer_key = 'I2n5GFnN525u7RX7p6Xdwf3jS'
    consumer_secret = 'UluAyPlYgNrh4LlmztjykWAWxSLjcZeEwDvhNEJGyi1y7kBpoF'
    access_token = '902925578159788032-60gpq6uMlVsEOAKwI98fijXYxKpfrYV'
    access_token_secret = 'hZN5PWUdJmO6MTmBBsx4WKuvuTx0Ve9aZlEGxvV1kOKfs'

    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    client = tweepy.API(auth, wait_on_rate_limit=True)

    for status in tweepy.Cursor(client.user_timeline, id=user).items(count):
        print(status.text)
        print(pred(status.text, classifier))

        if ((pred(status.text, classifier)) == 'Positive'):
            pcount = pcount + 1

        if ((pred(status.text, classifier)) == 'Negative'):
            ncount = ncount + 1

        pos_per = 100 * pcount / count
        neg_per = 100 * ncount / count

        if (pcount > ncount):
            analysis = user,'is positive minded'
        else:
            analysis = user,"is negative minded"

    return render(request, "predict2.html", {'result' : ncount,'ress':pcount, 'pp':pos_per, 'np':neg_per, 'analysis':analysis})






def pred(tweet, classifier):
    tweet_processed = stem(preprocessTweets(tweet))

    if (('__positive__') in (tweet_processed)):
        sentiment = 1
        return sentiment

    elif (('__negative__') in (tweet_processed)):
        sentiment = 0
        return sentiment
    else:

        X = [tweet_processed]
        sentiment = classifier.predict(X)
        return (sentiment[0])


def preprocessTweets(tweet):
    # Convert www.* or https?://* to URL
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL', tweet)

    # Convert @username to __HANDLE
    tweet = re.sub('@[^\s]+', '__HANDLE', tweet)

    # Replace #word with word
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)

    # trim
    tweet = tweet.strip('\'"')

    # Repeating words like happyyyyyyyy
    rpt_regex = re.compile(r"(.)\1{1,}", re.IGNORECASE)
    tweet = rpt_regex.sub(r"\1\1", tweet)

    # Emoticons
    emoticons = \
        [
            ('__positive__', [':-)', ':)', '(:', '(-:', \
                              ':-D', ':D', 'X-D', 'XD', 'xD', \
                              '<3', ':\*', ';-)', ';)', ';-D', ';D', '(;', '(-;', ]), \
            ('__negative__', [':-(', ':(', '(:', '(-:', ':,(', \
                              ':\'(', ':"(', ':((', ]), \
            ]

    def replace_parenth(arr):
        return [text.replace(')', '[)}\]]').replace('(', '[({\[]') for text in arr]

    def regex_join(arr):
        return '(' + '|'.join(arr) + ')'

    emoticons_regex = [(repl, re.compile(regex_join(replace_parenth(regx)))) \
                       for (repl, regx) in emoticons]

    for (repl, regx) in emoticons_regex:
        tweet = re.sub(regx, ' ' + repl + ' ', tweet)

    # Convert to lower case
    tweet = tweet.lower()

    return tweet


# Stemming of Tweets

def stem(tweet):
    stemmer = nltk.stem.PorterStemmer()
    tweet_stem = ''
    words = [word if (word[0:2] == '__') else word.lower() \
             for word in tweet.split() \
             if len(word) >= 3]
    words = [stemmer.stem(w) for w in words]
    tweet_stem = ' '.join(words)
    return tweet_stem



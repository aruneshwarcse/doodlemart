from flask import Flask, render_template, request, redirect, url_for, session 
from flask_mysqldb import MySQL 
import MySQLdb.cursors 
import sys,tweepy,csv,re
import speech_recognition
from textblob import TextBlob
import os
import smtplib
import argparse
from paz.backend.camera import VideoPlayer
from paz.backend.camera import Camera
from paz.pipelines import DetectMiniXceptionFER
from fer import FER
import cv2
import pprint
from werkzeug.utils import secure_filename
from numpy import asarray
import numpy as np
from werkzeug.datastructures import FileStorage
from nltk.corpus import twitter_samples
from nltk.tag import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
import re, string
from nltk.corpus import stopwords
import random
from nltk import classify,NaiveBayesClassifier,FreqDist
from nltk.tokenize import word_tokenize

app = Flask(__name__)
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

app.secret_key = 'abc123lkj'
  
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = os.environ.get('MYSQL_PASS')
app.config['MYSQL_DB'] = 'doodlemart'

mysql = MySQL(app)  

@app.route('/')
@app.route('/login',methods=['GET','POST']) 
def login():
    #login process here 
    global email
    msg=''
    if request.method == 'POST' and 'email' in request.form and 'password' in request.form: 
        # global username
        email = request.form['email'] 
        password = request.form['password'] 
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor) #for traversing the results
        cursor.execute('SELECT * FROM accounts WHERE email = % s AND password = % s', (email, password, )) 
        account = cursor.fetchone()         
        if email == "admin" and password == "ad123":
            return render_template('index.html')

        if account: 
            session['loggedin'] = True
            session['id'] = account['id'] 
            session['email'] = account['email'] 
            msg = 'Logged in successfully !'
            return render_template('home.html') 
        else: 
            msg = 'Incorrect username / password !'
        
    return render_template('login.html', msg = msg) 

@app.route('/register', methods =['GET', 'POST']) 
def register(): 
    #register process below
    msg = '' 
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form and 'email' in request.form : 
        username = request.form['username'] 
        password = request.form['password'] 
        email = request.form['email'] 
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor) 
        cursor.execute('SELECT * FROM accounts WHERE email = % s', (email, )) 
        account = cursor.fetchone() 
        if account: 
            msg = 'Account already exists !'
        elif not re.match(r'[^@]+@[^@]+\.[^@]+', email): 
            msg = 'Invalid email address !'
        elif not re.match(r'[A-Za-z0-9]+', username): 
            msg = 'Username must contain only characters and numbers !'
        elif not username or not password or not email: 
            msg = 'Please fill out the form !'
        else: 
            cursor.execute('INSERT INTO accounts VALUES (NULL, % s, % s, % s)', (username, password, email, )) 
            mysql.connection.commit() 
            msg = 'You have successfully registered !'
    elif request.method == 'POST': 
        msg = 'Please fill out the form !'
    return render_template('register.html', msg = msg) 


@app.route('/logout') 
def logout(): 
    session.pop('loggedin', None) 
    session.pop('id', None) 
    session.pop('email', None) 
    EMAIL_ADDRESS = os.environ.get('EMAIL_USER')
    EMAIL_PASSWORD = os.environ.get('EMAIL_PASS')
    #uijdjvprzgnlrymy

    with smtplib.SMTP('smtp.gmail.com',587) as smtp:
        smtp.ehlo()
        smtp.starttls()
        smtp.ehlo()
        smtp.login(EMAIL_ADDRESS,EMAIL_PASSWORD)
        subject='Dear Aruneshwar!'
        body='Thank you for visiting Doodlemart.We hope that you will return soon :) '

        msg=f'Subject:{subject}\n\n{body}'
        smtp.sendmail(EMAIL_ADDRESS,email,msg)

    return redirect(url_for('login'))

    
@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/cart')
def cart():
    return render_template('cart.html')

@app.route('/category')
def category():
    return render_template('category.html')

@app.route('/checkout')
def checkout():
    return render_template('checkout.html')

@app.route('/confirmation')
def confirmation():
    return render_template('confirmation.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/tracking')
def tracking():
    return render_template('tracking.html')

@app.route('/watch6')
def watch6():
    return render_template('watch6.html')


@app.route('/ipad')
def ipad():
    return render_template('ipad.html')

#chart for ipad
@app.route('/chart',methods=['GET','POST'])
def chart():
    msg=''
    tweets=[]
    tweetText=[]
    consumerKey = os.environ.get('consumerKey')
    consumerSecret = os.environ.get('consumerSecret')
    accessToken = os.environ.get('accessToken')
    accessTokenSecret = os.environ.get('accessTokenSecret')
    
    def cleanTweet(tweet):
        return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t]) | (\w +:\ / \ / \S +)", " ", tweet).split())

    auth = tweepy.OAuthHandler(consumerKey, consumerSecret)
    auth.set_access_token(accessToken, accessTokenSecret)
    api = tweepy.API(auth)

    searchTerm ="ipad"
    NoOfTerms = 100
    
    tweets = tweepy.Cursor(api.search, q=searchTerm, lang = "en").items(NoOfTerms)
    polarity = 0
    positive = 0
    negative = 0

    for tweet in tweets:
        tweetText.append(cleanTweet(tweet.text).encode('utf-8'))
        analysis = TextBlob(tweet.text)
        polarity += analysis.sentiment.polarity 

        if (analysis.sentiment.polarity >=0.00):
            positive += 1
        elif (analysis.sentiment.polarity < 0.00):
            negative += 1

    if (polarity > 0.00):
        msg="pos"
        print("Positive")
        
    elif (polarity <= -0.00):
        print("Negative")
        msg="neg"
    return render_template('chart.html',pos=positive,neg=negative)

@app.route('/harddisk')
def harddisk():
    return render_template('harddisk.html')

#chart for harddisk
@app.route('/hddchart',methods=['GET','POST'])
def hddchart():
    msg=''
    tweets=[]
    tweetText=[]
    consumerKey = os.environ.get('consumerKey')
    consumerSecret = os.environ.get('consumerSecret')
    accessToken = os.environ.get('accessToken')
    accessTokenSecret = os.environ.get('accessTokenSecret')
    def cleanTweet(tweet):
        return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t]) | (\w +:\ / \ / \S +)", " ", tweet).split())

    auth = tweepy.OAuthHandler(consumerKey, consumerSecret)
    auth.set_access_token(accessToken, accessTokenSecret)
    api = tweepy.API(auth)

    searchTerm ="seagate"
    NoOfTerms = 100
    
    tweets = tweepy.Cursor(api.search, q=searchTerm, lang = "en").items(NoOfTerms)
    polarity = 0
    positive = 0
    negative = 0

    for tweet in tweets:
        tweetText.append(cleanTweet(tweet.text).encode('utf-8'))
        analysis = TextBlob(tweet.text)
        polarity += analysis.sentiment.polarity 

        if (analysis.sentiment.polarity >=0.00):
            positive += 1
        elif (analysis.sentiment.polarity < 0.00):
            negative += 1

    if (polarity > 0.00):
        msg="pos"
        print("Positive")
        
    elif (polarity <= -0.00):
        print("Negative")
        msg="neg"
    return render_template('hddchart.html',pos=positive,neg=negative)

@app.route('/echostudio')
def echostudio():
    return render_template('echostudio.html')

#chart for amazon echo
@app.route('/echochart',methods=['GET','POST'])
def echochart():
    msg=''
    tweets=[]
    tweetText=[]
    consumerKey = os.environ.get('consumerKey')
    consumerSecret = os.environ.get('consumerSecret')
    accessToken = os.environ.get('accessToken')
    accessTokenSecret = os.environ.get('accessTokenSecret')
    
    def cleanTweet(tweet):
        return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t]) | (\w +:\ / \ / \S +)", " ", tweet).split())

    auth = tweepy.OAuthHandler(consumerKey, consumerSecret)
    auth.set_access_token(accessToken, accessTokenSecret)
    api = tweepy.API(auth)

    searchTerm ="amazonecho"
    NoOfTerms = 100
    
    tweets = tweepy.Cursor(api.search, q=searchTerm, lang = "en").items(NoOfTerms)
    polarity = 0
    positive = 0
    negative = 0

    for tweet in tweets:
        tweetText.append(cleanTweet(tweet.text).encode('utf-8'))
        analysis = TextBlob(tweet.text)
        polarity += analysis.sentiment.polarity 

        if (analysis.sentiment.polarity >=0.00):
            positive += 1
        elif (analysis.sentiment.polarity < 0.00):
            negative += 1

    if (polarity > 0.00):
        msg="pos"
        print("Positive")
        
    elif (polarity <= -0.00):
        print("Negative")
        msg="neg"
    return render_template('echochart.html',pos=positive,neg=negative)

@app.route('/iphonese')
def iphonese():
    return render_template('iphonese.html')

#chart for iphone SE
@app.route('/iphonechart',methods=['GET','POST'])
def iphonechart():
    msg=''
    tweets=[]
    tweetText=[]
    consumerKey = os.environ.get('consumerKey')
    consumerSecret = os.environ.get('consumerSecret')
    accessToken = os.environ.get('accessToken')
    accessTokenSecret = os.environ.get('accessTokenSecret')
    def cleanTweet(tweet):
        return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t]) | (\w +:\ / \ / \S +)", " ", tweet).split())

    auth = tweepy.OAuthHandler(consumerKey, consumerSecret)
    auth.set_access_token(accessToken, accessTokenSecret)
    api = tweepy.API(auth)

    searchTerm ="iphone"
    NoOfTerms = 100
    
    tweets = tweepy.Cursor(api.search, q=searchTerm, lang = "en").items(NoOfTerms)
    polarity = 0
    positive = 0
    negative = 0

    for tweet in tweets:
        tweetText.append(cleanTweet(tweet.text).encode('utf-8'))
        analysis = TextBlob(tweet.text)
        polarity += analysis.sentiment.polarity 

        if (analysis.sentiment.polarity >=0.00):
            positive += 1
        elif (analysis.sentiment.polarity < 0.00):
            negative += 1

    if (polarity > 0.00):
        msg="pos"
        print("Positive")
        
    elif (polarity <= -0.00):
        print("Negative")
        msg="neg"
    return render_template('iphonechart.html',pos=positive,neg=negative)



@app.route('/voice',methods=['GET','POST'])
def voice():
    return render_template('voice.html')


@app.route('/voice2',methods=['GET','POST'])
def voice2():
    print("Ki")
    a=''
    pos=''
    recognizer = speech_recognition.Recognizer()
    with speech_recognition.Microphone() as source:
        print("Say something!")
        audio = recognizer.listen(source)

    print("You said:")
    a=recognizer.recognize_google(audio)

    print(a)
    y=a
    edu=TextBlob(y)
    x=edu.sentiment.polarity

    if x<0:
        pos="Negative"
        print("Negative")

    elif x>=0 and x<=1:
        pos="positive"
        print("positive")
    return render_template('voice333.html',a=a,pos=pos)


@app.route('/camera')
def camera():
    print('working')
    parser = argparse.ArgumentParser(description='Real-time face classifier')
    parser.add_argument('-c', '--camera_id', type=int, default=0,
                        help='Camera device ID')
    parser.add_argument('-o', '--offset', type=float, default=0.1,
                        help='Scaled offset to be added to bounding boxes')
    args = parser.parse_args()
    print('flyjad7it6ra7du')
    pipeline = DetectMiniXceptionFER([args.offset, args.offset])
    camera = Camera(args.camera_id)
    player = VideoPlayer((640, 480), pipeline, camera)
    print('cup ')
    player.run()
    
    return render_template('index.html')


@app.route('/images',methods=['GET','POST'])
def images():

    return render_template('imagesentiment.html')

@app.route('/imagechart',methods=['GET','POST'])
def imagechart():
    if request.method=="POST":
        image=request.files['file']
        # print(type(image))
        # print(image.filename)
        imagee = cv2.imread(image.filename)
        detector = FER()
        result=detector.detect_emotions(imagee)
        ans = result[0]
        # pprint.pprint(result)
        emotion, score = detector.top_emotion(imagee) 
        # pprint.pprint(emotion)
        # pprint.pprint(score)
        final = ans["emotions"]
        angry = final["angry"]
        disgust = final["disgust"]
        fear = final["fear"]
        happy = final["happy"]
        sad = final["sad"]
        surprise = final["surprise"]
        neutral = final["neutral"]
    return render_template('imagechart.html',angry=angry,disgust=disgust,fear=fear,happy=happy,sad=sad,surprise=surprise,neutral=neutral)

@app.route('/textsentiment')
def textsentiment():
    return render_template('textsentiment.html')

@app.route('/textsentimentans',methods=['GET','POST'])
def textsentimentans():
    v1=''
    result=''
    if request.method=="POST":
        v1=request.form['textinput']
    
    # print(v1)
    positive_tweets = twitter_samples.strings('positive_tweets.json')
    negative_tweets = twitter_samples.strings('negative_tweets.json')
    text = twitter_samples.strings('tweets.20150430-223406.json')
    tweet_tokens = twitter_samples.tokenized('positive_tweets.json')

    def lemmatize_sentence(tokens):
        lemmatizer = WordNetLemmatizer()
        lemmatized_sentence = []
        for word, tag in pos_tag(tokens):
            if tag.startswith('NN'):
                pos = 'n'
            elif tag.startswith('VB'):
                pos = 'v'
            else:
                pos = 'a'
            lemmatized_sentence.append(lemmatizer.lemmatize(word, pos))
        return lemmatized_sentence

    def remove_noise(tweet_tokens, stop_words = ()):
        cleaned_tokens = []
        for token, tag in pos_tag(tweet_tokens):
            token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'\
                        '(?:%[0-9a-fA-F][0-9a-fA-F]))+','', token)
            token = re.sub("(@[A-Za-z0-9_]+)","", token)

            if tag.startswith("NN"):
                pos = 'n'
            elif tag.startswith('VB'):
                pos = 'v'
            else:
                pos = 'a'

            lemmatizer = WordNetLemmatizer()
            token = lemmatizer.lemmatize(token, pos)

            if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
                cleaned_tokens.append(token.lower())
        return cleaned_tokens

    stop_words = stopwords.words('english')
    positive_tweet_tokens = twitter_samples.tokenized('positive_tweets.json')
    negative_tweet_tokens = twitter_samples.tokenized('negative_tweets.json')
    positive_cleaned_tokens_list = []
    negative_cleaned_tokens_list = []

    for tokens in positive_tweet_tokens:
        positive_cleaned_tokens_list.append(remove_noise(tokens, stop_words))

    for tokens in negative_tweet_tokens:
        negative_cleaned_tokens_list.append(remove_noise(tokens, stop_words))

    def get_all_words(cleaned_tokens_list):
        for tokens in cleaned_tokens_list:
            for token in tokens:
                yield token
    
    all_pos_words = get_all_words(positive_cleaned_tokens_list)

    freqdistpos = FreqDist(all_pos_words)

    def get_tweets_for_model(cleaned_tokens_list):
        for tweet_tokens in cleaned_tokens_list:
            yield dict([token, True] for token in tweet_tokens)
    
    positive_tokens_for_model = get_tweets_for_model(positive_cleaned_tokens_list)
    negative_tokens_for_model = get_tweets_for_model(negative_cleaned_tokens_list)

    posData = [(tweet_dict, "Positive")
                     for tweet_dict in positive_tokens_for_model]

    negData = [(tweet_dict, "Negative")
                     for tweet_dict in negative_tokens_for_model]
    
    dataset = posData + negData

    random.shuffle(dataset)
    train_data = dataset[:7000]
    test_data = dataset[7000:]
    classifier = NaiveBayesClassifier.train(train_data)
    print("input")
    textInput = v1
    custom_tokens = remove_noise(word_tokenize(textInput))
    print(classifier.classify(dict([token, True] for token in custom_tokens)))
    result=classifier.classify(dict([token, True] for token in custom_tokens))
    return render_template('textsentiment.html',result=result)

if __name__ == "__main__":
    app.run(debug=True)
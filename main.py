# server libraries
from flask import Flask
from flask import request
from flask_cors import CORS

# nlp libraries
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import numpy as np

# python libraries
import io
import random
import string
import warnings
import pickle
import re
import datetime as dt
import json
import os

# download nltk libraries
nltk.download('all')

class User:
    """Defines user class to remember user, likes, dislikes"""

    def __init__(self, n, age, sch, classify): # name, age, school user went to, and classfication of user
        self.name = n.lower()
        self.age = age
        self.school = sch
        self.classify = classify
        self.likes = []
        self.dislikes = []
        self.neutral = []
        self.lastContact = dt.datetime.now()

    def getName(self):
        return self.name

    def getAge(self):
        return self.age

    def getSchool(self):
        return self.school

    def getClass(self):
        return self.classify

    def getLikes(self): # returns a random like, else none
        if len(self.likes) > 0:
            return random.choice(self.likes)
        else:
            return None

    def getDislikes(self): # returns a random dislike, else none
        if len(self.dislikes) > 0:
            return random.choice(self.dislikes)
        else:
            return None

    def getNeutral(self): # returns a random neutral, else none
        if len(self.neutral) > 0:
            return random.choice(self.neutral)
        else:
            return None

    def addPreference(self, sent): # Uses sentiment analysis to add a sentence to a user's like/dislike/or neutral
        analyzer = SentimentIntensityAnalyzer()
        score = analyzer.polarity_scores(sent)
        if score['pos'] >= score['neu'] and score['pos'] >= score['neg']:
            self.likes.append(sent)
        elif score['neg'] >= score['neu'] and score['neg'] >= score['pos']:
            self.dislikes.append(sent)
        else:
            self.neutral.append(sent)

    def getDaysFromLastContact(self):
        return (dt.datetime.now() - self.lastContact).days

    def updateLastContact(self):
        self.lastContact = dt.datetime.now()

    def display(self): ## debug output
        print("Name:", self.getName())
        print("\tAge:", self.getAge())
        print("\tSchool:", self.getSchool())
        print("\tGrade:", self.getClass())
        if self.getLikes() is not None:
            print("\tLikes:", str(self.likes))
        else:
            print("\tLikes: <None>")
        if self.getDislikes() is not None:
            print("\tDislikes:", str(self.dislikes))
        else:
            print("\tDislikes: <none>")
        if self.getNeutral() is not None:
            print("\tNeutral:", str(self.neutral))
        else:
            print("\tNeutral: <none>")
# End class user definition


### User Class Interface ###
def createNewUser(kb, name, age, school, classification):
    kb[name] = User(name, age, school, classification)

def addSentenceToUserPref(kb, name, sent):
    kb[name].addPreference(sent)

def closeUserSession(kb, name):
    kb[name].updateLastContact()
    save_users(kb)

def printUsers(users): # for debugging purposes only
      print("\nDump of users in User database:")
      for n in users.keys():
            users[n].display()

#### Pickling / Loading ####

# Load or Create users dict
def load_users():
    # if users exist, load users. Otherwise create users (first use)
    if os.path.isfile("users.p"):
        users = unpickle_users()
    else:
        users = {}
    return users

# Read in user objects
def unpickle_users():
    with open("users.p", 'rb') as pf:
        print("STATUS: Loaded users.p")
        return pickle.load(pf)

# if newton.kb exists, load kb. Otherwise create from scrapped files.
def load_tokens():
    if os.path.isfile("newton_kb.p"):
        s_tokens = unpickle_kb()
    else:
        s_tokens = import_raw_data(
        )  # read and tokenize scrapped data from HW05
    return s_tokens

# Read in kb
def unpickle_kb():
    with open("newton_kb.p", 'rb') as pf:
        print("STATUS: loaded newton_kb.p")
        return pickle.load(pf)

# Save user objects
def save_users(users):
    with open("users.p", 'wb') as pf:
        print("STATUS: Saved users.p")
        pickle.dump(users, pf)
    
# Read in scrapped files (from Homework5)
def import_raw_data(pathName):
    doc = pathName  # base filename
    docs = ""  # list of text
    num_docs = 7
    for i in range(1, num_docs + 1):
        file = doc + str(i) + '.txt'
        with open(file, 'r', encoding='utf-8') as f:
            raw_text = f.read().lower()  # lowercase
            raw_text = re.sub("[\(\[].*?[\)\]]", "", raw_text)  # remove refs
            docs += raw_text.replace('\n', ' ')  # remove newline
    raw = docs
    # Tokenized Sentences
    sents = nltk.sent_tokenize(raw)
    # Pickle kb for future use
    with open("newton_kb.p", 'wb') as pf:
        print("STATUS: Saved newton_kb.p")
        pickle.dump(sents, pf)
    return sents

# Preprocessing functions
def lemTokens(tokens):
    wnl = WordNetLemmatizer()
    return [wnl.lemmatize(token) for token in tokens]


def lemNormalize(text):
    remove_punct_dict = dict(
        (ord(punct), None) for punct in string.punctuation)
    lemTkn = lemTokens(
        nltk.word_tokenize(text.lower().translate(remove_punct_dict)))
    return lemTkn

# Generating response to user sentence
def response(sentence, kb):
    """If user's input is not a greeting, return an appropriate response from kb"""
    bot_response = ''
    kb.append(sentence)  # add user_response to kb for vectorization
    # convert raw docs to matrix of TF-IDF features
    tfidfVectorize = TfidfVectorizer(
        tokenizer=lemNormalize, stop_words='english')
    tfidf = tfidfVectorize.fit_transform(
        sent_tokens)  # learn vocab & idf, return term matrix
    vals = cosine_similarity(tfidf[-1],
                             tfidf)  # measure similarity w.r.t. user input
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    # check most similar score
    if req_tfidf == 0:
        bot_response = bot_response + "I am sorry! I can't comment on that."
    else:
        bot_response = bot_response + sent_tokens[idx]
    kb.remove(sentence)  # restore kb
    return bot_response

#### Knowledgebase Helper Methods ####

def getRandomResponse():
    return sent_tokens[random.randint(0, len(sent_tokens))]

def getRandomUserFact(user):
  returnArr = [f"Hmm... {user.getAge()} is such a great age!", f"Hmm... I've been hearing a lot of great things from {user.getSchool()}!", f"Hmm... I bet you cannot wait for your {user.getClass().lower()} year to be over!"]

  return random.choice(returnArr)

#### JSON HELPER METHODS #### (For Webhook Interface)
def getTextJSONResponse(text):
    return {'fulfillment_response': {'messages': [{'text': {'text': [text]}}]}}

def getUserInfoJSONResponse(user, text):
    return {
        'fulfillment_response': {
            'messages': [{
                'text': {
                    'text': [text]
                }
            }]
        },
        "session_info": {
            "parameters": {
                "userage": user.getAge(),
                "userclassification": user.getClass(),
                "userschool": user.getSchool(),
                "isPreviousUser": True
            }
        }
    }

def getUserNotFoundJSONResponse(name):
    return {
        'fulfillment_response': {
            'messages': [{
                'text': {
                    'text': [f"Nice to meet you {name}!"]
                }
            }]
        },
        "session_info": {
            "parameters": {
                "isPreviousUser": False
            }
        }
    }

def getTagType(jsonQuery):
    return jsonQuery['fulfillmentInfo']['tag']

#### FLASK APPLICATION ####

users = load_users() # user knowledgebase

sent_tokens = load_tokens() # newton information knowledgebase
app = Flask('app') # opens flask app
CORS(app)

@app.route('/', methods=['GET'])
def keepAlive():
  return "Active"

@app.route('/', methods=['POST'])
def routeIntent():
    req = request.get_json() # gets JSON from POST request
    intent = getTagType(req) # gets intent of request

    if intent == 'getRandomFact':
        userName = req["sessionInfo"]['parameters']['given-name'].lower()
        print(f"Returning random fact to user: {userName}")
        return getTextJSONResponse(getRandomResponse())

    elif intent == 'askQuestionEntityFacts': # user asks question, algorithm will respond
        userName = req["sessionInfo"]['parameters']['given-name'].lower()
        question = req["intentInfo"]['parameters']['message']['originalValue']
        if question[-1] == '?': # remove question mark at end of sentence
            question = question[:-1]
        textResp = getTextJSONResponse(response(question, sent_tokens))
        print(f"Responding to user {userName} question: {question}")
        return textResp

    elif intent == 'askQuestionUserTalk': # user asks question, algorithm will respond
        userName = req["sessionInfo"]['parameters']['given-name'].lower()
        question = req["intentInfo"]['parameters']['message1']['originalValue']
        if question[-1] == '?': # remove question mark at end of sentence
            question = question[:-1]
        textResp = getTextJSONResponse(response(question, sent_tokens))
        print(f"Responding to user {userName} question: {question}")
        return textResp

    elif intent == 'checkUserInKB':
        userName = req["intentInfo"]['parameters']['given-name']['originalValue']
        print(f"Checking {userName} in user knowledgebase...")
        if userName.lower() in users: # if user exists in knowledgebase
          userObj = users[userName.lower()]
          userObj.updateLastContact()
          print(f"User {userName} exists in user knowledgebase")
          return getUserInfoJSONResponse(userObj, f"Welcome back {userName}! {getRandomUserFact(userObj)} You haven't grown an inch since I last saw you {userObj.getDaysFromLastContact()} days ago.")
        else: # else user does not exist
          print(f"User {userName} does not exist in user knowledgebase")
          return getUserNotFoundJSONResponse(userName)

    elif intent == 'addNewUser':
        userName = req["sessionInfo"]['parameters']['given-name'].lower()
        userAge = int(req["sessionInfo"]['parameters']['userage'])
        userClass = req["sessionInfo"]['parameters']['userclassification']
        userSchool = req["sessionInfo"]['parameters']['userschool']
        createNewUser(users, userName, userAge, userSchool, userClass)
        print(f"User {userName} created with params (age: {userAge}, school: {userSchool}, classification: {userClass})")
        save_users(users) # saves user file
        return getTextJSONResponse("")

    elif intent == "blanketStatement": # adds a sentence to user's likes/dislikes/neutral
        userName = req["sessionInfo"]['parameters']['given-name'].lower()
        sent = req['intentInfo']['parameters']['blanketsent']['originalValue']
        addSentenceToUserPref(users, userName, sent)
        save_users(users) # saves user file
        print(f"Added \"{sent}\" to user {userName} preferences")
        return getTextJSONResponse("")

    elif intent == 'lastConvo': # get a random like/dislike/neutral from a user
        userName = req["sessionInfo"]['parameters']['given-name'].lower()
        userObj = users[userName]
        
        sentimentList = []

        if userObj.getLikes():
          sentimentList.append(('like', userObj.getLikes()))

        if userObj.getDislikes():
          sentimentList.append(('dislike', userObj.getDislikes()))

        if userObj.getNeutral():
          sentimentList.append(('neutral', userObj.getNeutral()))

        if len(sentimentList) == 0:
            return getTextJSONResponse("")
  
        randomChoice, message = random.choice(sentimentList)

        if randomChoice == 'like':
          return getTextJSONResponse(f"You really liked Newton the last time we talked when you said \"{message}\"")
        elif randomChoice == 'dislike':
          return getTextJSONResponse(f"I remember you disliked Newton when you said \"{message}\"")
        elif randomChoice == 'neutral':
          return getTextJSONResponse(f"You made a really good point last time when you said \"{message}\"")
        else:
          return getTextJSONResponse("")
          
    elif intent == "quit":
        userName = req["sessionInfo"]['parameters']['given-name'].lower()
        closeUserSession(users, userName)
        print(f"{userName} has quit and session has been saved")
        return getTextJSONResponse("")

    return getTextJSONResponse("Server Error (Perhaps no intent mapping?)")

app.run(host='0.0.0.0', port=8080)
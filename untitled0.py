import nltk
import random
import string
import numpy as np

f=open('corpus.txt','r',errors= 'ignore')
raw=f.read()
raw=raw.lower()

sent_token = nltk.sent_tokenize(raw)
word_token= nltk.word_tokenize(raw)

lemmer=nltk.stem.WordNetLemmatizer()

def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]

remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

Greeting_Inputs=("hello", "hi", "greetings", "what's up" ,"sup","hey","how are you")
Greeting_Responses=("hello", "hi there", "greetings", "I am glad!","You are talking to me","hey","fuck you")

def greeting(sentence):
    for word in sentence.split():
        if word.lower() in Greeting_Inputs:
            return random.choice(Greeting_Responses)
        
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def response(user_response):
    robo_response = ''
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_token)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx=vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    
    if(req_tfidf==0):
        robo_response=robo_response+"I am sorry! I don't understand you"
        return robo_response
    else:
        robo_response = robo_response+sent_token[idx]
        return robo_response
    
flag=True
print("ROBO: My name is Robo. I will answer your queries about Chatbots. If you want to exit, type Bye!")
while(flag==True):
    user_response = input()
    user_response=user_response.lower()
    if(user_response!='bye'):
        if(user_response=='thanks' or user_response=='thank you' ):
            flag=False
            print("ROBO: You are welcome..")
        else:
            if(greeting(user_response)!=None):
                print("ROBO: "+greeting(user_response))
            else:
                sent_token.append(user_response)
                word_token=word_token+nltk.word_tokenize(user_response)
                final_words=list(set(word_token))
                print("ROBO: ",end="")
                print(response(user_response))
                sent_token.remove(user_response)
    else:
        flag=False
        print("ROBO: Bye! take care..")
    
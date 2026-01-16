from flask import Flask, request, jsonify
import random
import string
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import WordNetLemmatizer

app = Flask(__name__)

# --- INITIALIZE CHATBOT DATA ---
nltk.download('popular', quiet=True)
nltk.download('punkt_tab', quiet=True)

# Global variables
lemmer = WordNetLemmatizer()
sent_tokens = []

try:
    with open('chatbot.txt', 'r', encoding='utf8', errors='ignore') as fin:
        raw = fin.read().lower()
    sent_tokens = nltk.sent_tokenize(raw)
except Exception as e:
    print(f"Error loading file: {e}")

def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]

remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up", "hey")
GREETING_RESPONSES = ["hi", "hey", "*nods*", "hi there", "hello", "I am glad! You are talking to me"]

def greeting(sentence):
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)

def get_robo_response(user_response):
    robo_response = ''
    sent_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    
    if req_tfidf == 0:
        robo_response = "I am sorry! I don't understand you"
    else:
        robo_response = sent_tokens[idx]
    
    sent_tokens.remove(user_response) # Clean up after response
    return robo_response

# --- FLASK ROUTE ---
@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        user_msg = data.get("message", "")
        
        # Check for greetings first
        if greeting(user_msg) is not None:
            reply = greeting(user_msg)
        else:
            reply = get_robo_response(user_msg)
            
        return jsonify({"reply": reply})
    except Exception as e:
        print(f"CRASH: {e}") # This prints the error to your PC console
        return jsonify({"reply": "Python Error: " + str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
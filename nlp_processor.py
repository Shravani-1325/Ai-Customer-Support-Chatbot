import re
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

class NLPProcessor:
  def __init__(self):
    self.lemmatizer = WordNetLemmatizer()
    self.stop_words = set(stopwords.words("english"))
    self.stop_words.discard("no")
    self.stop_words.discard("not")

  def clean_text(self, x):
    x =  x.lower()
    
    # Punctuation 
    for i in x:
      if i in string.punctuation:
        x = x.replace(i, "")
    x = "".join([char for char in x if not char.isdigit()])
    x = " ".join(x.split()) 

    return x

  def preprocess_text(self, x):
    x = str(x)
    
    # Tokenization 
    words = word_tokenize(x) 

    # Stopswords + Lemmatization
    cleaned_words = []

    for word in words:
      if word not in self.stop_words:
        lem = self.lemmatizer.lemmatize(word) 
        cleaned_words.append(lem)
        s = " ".join(cleaned_words)

    return s

  def main(self, text):
    x = self.clean_text(text)
    token = self.preprocess_text(x)

    return token
  

processor = NLPProcessor()

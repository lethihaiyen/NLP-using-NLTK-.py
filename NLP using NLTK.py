#!/usr/bin/env python
# coding: utf-8

# ### Natural language processing

# Natural Language Processing : 
#     
# - Linguistics + Data Science + Machine Learning Model 
# 
# - branch of artificial intelligence to decipher language structures 
# 
# Segmentation : break out each sentences by full stops.
#     
# Tokenization : each word is a token. 
#     
# Stop words : remove meaningless words. 
#     
# Stemming : normalize words to the basic root ( wait, waiting -> wait ) 
#     
# Lemmatization : it's the same as stemming but will generate an actual words ( sometimes stemming generate false words  by cutting the end/start prefix of a words. ) 
#     
# Part of speech (POS) tagging : add noun-verb-adj tag to the words. 
#     
# Name entity recognition (NER) : catergorized name, geographical, monument tag without human analysis. 
# 

# In[1]:


text = "Millions of people across the UK and beyond have celebrated the coronation of King Charles III - a symbolic ceremony combining a religious service and pageantry. The ceremony was held at Westminster Abbey, with the King becoming the 40th reigning monarch to be crowned there since 1066. Queen Camilla was crowned alongside him before a huge parade back to Buckingham Palace. Here's how the day of splendour and formality, which featured customs dating back more than 1,000 years, unfolded."
text


# #### Segmentation

# In[2]:


# import 
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize


# In[3]:


# Split text into sentences
sentences = sent_tokenize(text)
sentences


# In[4]:


#indexing sentences  :
sentences[2]


# In[5]:


# Punctuation removal
import re

# Remove punctuation characters - remove the dot 
text = re.sub(r"[^a-zA-Z0-9]", " ", sentences[2]) 
text


# #### Tokenization

# In[6]:


from nltk.tokenize import word_tokenize


# In[7]:


words = word_tokenize(text)
print(words)


# #### Removal of stop words

# In[8]:


nltk.download('stopwords')
from nltk.corpus import stopwords


# In[9]:


# Remove stop words
words = [w for w in words if w not in stopwords.words("english")]
print(words)


# In[25]:


# have a look at the stop words in nltk's corpus
print(stopwords.words("english"))


# #### Stemming and lemmatization

# In[11]:


nltk.download('wordnet') # download for lemmatization
nltk.download('omw-1.4')


# In[12]:


# Stemming
from nltk.stem.porter import PorterStemmer

# Reduce words to their stems
stemmed = [PorterStemmer().stem(w) for w in words]
print(stemmed)


# In[13]:


# Lemmatize
from nltk.stem.wordnet import WordNetLemmatizer

# Reduce words to their root form
lemmatized = [WordNetLemmatizer().lemmatize(w) for w in words]
print(lemmatized)


# In[14]:


# Another stemming and lemmatization example
words2 = ['wait', 'waiting' , 'studies', 'studying', 'computers']

# Stemming
# Reduce words to their stems
stemmed = [PorterStemmer().stem(w) for w in words2]
print("Stemming output: {}".format(stemmed))

# Lemmatization
# Reduce words to their root form
lemmatized = [WordNetLemmatizer().lemmatize(w) for w in words2]
print("Lemmatization output: {}".format(lemmatized))


# #### Part of speech tagging

# In[15]:


nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')


# In[16]:


from nltk import pos_tag


# In[17]:


# tag each word with part of speech
pos_tag(words)


# In[18]:


"""
POS

CC: It is the conjunction of coordinating
CD: It is a digit of cardinal
DT: It is the determiner
EX: Existential
FW: It is a foreign word
IN: Preposition and conjunction
JJ: Adjective
JJR and JJS: Adjective and superlative
LS: List marker
MD: Modal
NN: Singular noun
NNS, NNP, NNPS: Proper and plural noun
PDT: Predeterminer
WRB: Adverb of wh
WP$: Possessive wh
WP: Pronoun of wh
WDT: Determiner of wp
VBZ: Verb
VBP, VBN, VBG, VBD, VB: Forms of verbs
UH: Interjection
TO: To go
RP: Particle
RBS, RB, RBR: Adverb
PRP, PRP$: Pronoun personal and professional

"""


# #### Named entity recognition

# In[19]:


from nltk import ne_chunk
nltk.download('words')


# In[20]:


#name entity recognition tree 
ner_tree = ne_chunk(pos_tag(word_tokenize(sentences[2])))
print(ner_tree)


# In[21]:


text = "Millions of people across the UK and beyond have celebrated the coronation of King Charles III - a symbolic ceremony combining a religious service and pageantry. The ceremony was held at Westminster Abbey, with the King becoming the 40th reigning monarch to be crowned there since 1066. Queen Camilla was crowned alongside him before a huge parade back to Buckingham Palace. Here's how the day of splendour and formality, which featured customs dating back more than 1,000 years, unfolded."

ner_tree = ne_chunk(pos_tag(word_tokenize(text)))
print(ner_tree)


# In[22]:


text = "Twitter CEO Elon Musk arrived at the Staples Center in Los Angeles, California. "
ner_tree = ne_chunk(pos_tag(word_tokenize(text)))
print(ner_tree)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





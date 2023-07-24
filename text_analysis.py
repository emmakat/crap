
import os
import re
import string
import numpy as np
import pandas as pd
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.util import ngrams
from textblob import TextBlob
from wordcloud import WordCloud
import warnings
import spacy
from spacy import displacy
from pathlib import Path
from spacy.matcher import PhraseMatcher, Matcher
from spacy.tokens import Span
import tempfile
import matplotlib.pyplot as plt
import seaborn as sns
import spacy_streamlit
from PIL import Image
from itertools import chain
import streamlit as st
# Download required NLTK resources
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from gensim.utils import tokenize
# Constants
STOPWORDS = stopwords.words('english')
punctuation = list(string.punctuation)
from nltk.tokenize import sent_tokenize
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.summarizers.lex_rank import LexRankSummarizer
# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d', '', text)
    text = re.sub(r'\n', '', text)
    text = text.strip()
    text = text.translate(str.maketrans('', '', string.punctuation))
    lemmatizer = WordNetLemmatizer()
    text = [lemmatizer.lemmatize(word) for word in text.split() if word not in STOPWORDS]
    text = ' '.join(text)
    return text

# Function to calculate word frequencies
def calculate_word_frequencies(doc, stopwords, punctuation):
    word_frequencies = {}
    for word in doc:
        if word.text.lower() not in stopwords:
            if word.text.lower() not in punctuation:
                if word.text not in word_frequencies.keys():
                    word_frequencies[word.text] = 1
                else:
                    word_frequencies[word.text] += 1
    return word_frequencies

# Function to calculate sentence scores
def calculate_sentence_scores(doc, word_frequencies):
    sentence_tokens = [sent for sent in doc.sents]
    sentence_score = {}
    for sent in sentence_tokens:
        for word in sent:
            if word.text.lower() in word_frequencies.keys():
                if sent not in sentence_score.keys():
                    sentence_score[sent] = word_frequencies[word.text.lower()]
                else:
                    sentence_score[sent] += word_frequencies[word.text.lower()]
    return sentence_score

# Create a word cloud function 
def create_wordcloud(text, image_path=None):
    st.write('Creating Word Cloud..')

    text = clean_text(text)

    if image_path is None:
        # Generate the word cloud
        wordcloud = WordCloud(width=600, height=600,
                              background_color='white',
                              stopwords=STOPWORDS,
                              min_font_size=10).generate(text)
    else:
        mask = np.array(Image.open(image_path))
        wordcloud = WordCloud(width=600, height=600,
                              background_color='white',
                              stopwords=STOPWORDS,
                              mask=mask,
                              min_font_size=5).generate(text)

    # plot the WordCloud image
    plt.figure(figsize=(8, 8), facecolor=None)
    plt.imshow(wordcloud, interpolation='nearest')
    plt.axis("off")
    plt.tight_layout(pad=0)

    st.pyplot()
    plt.show()  


# Function to plot the ngrams based on n and top k value
def plot_ngrams(text, n=2, topk=15):
    

    st.write('Creating N-Gram Plot..')

    text = clean_text(text)
    tokens = text.split()
    
    # get the ngrams 
    ngram_phrases = ngrams(tokens, n)
    
    # Get the most common ones 
    most_common = Counter(ngram_phrases).most_common(topk)
    
    # Make word and count lists 
    words, counts = [], []
    for phrase, count in most_common:
        word = ' '.join(phrase)
        words.append(word)
        counts.append(count)
    
    # Plot the barplot 
    plt.figure(figsize=(10, 6))
    title = "Most Common " + str(n) + "-grams in the text"
    plt.title(title)
    ax = plt.bar(words, counts)
    plt.xlabel("n-grams found in the text")
    plt.ylabel("Ngram frequencies")
    plt.xticks(rotation=90)
    plt.show()


def pos_tagger(s):
    
    # Define the tag dictionary 
    output = ''
    
    # Remove punctuations
    s = s.translate(str.maketrans('', '', string.punctuation))
    
    tagged_sentence = nltk.pos_tag(nltk.word_tokenize(s))
    for tag in tagged_sentence:
        out = tag[0] + ' ---> ' + tag[1] + '<br>'
        output += out

    return output
# Simple Regular expression (Regex) based NP(Noun phrase) based chunker
def generate_chunks(inp_str):
    max_chunk = 500
    inp_str = inp_str.replace('.', '.<eos>')
    inp_str = inp_str.replace('?', '?<eos>')
    inp_str = inp_str.replace('!', '!<eos>')

    sentences = inp_str.split('<eos>')
    current_chunk = 0
    chunks = []
    for sentence in sentences:
        if len(chunks) == current_chunk + 1:
            if len(chunks[current_chunk]) + len(sentence.split(' ')) <= max_chunk:
                chunks[current_chunk].extend(sentence.split(' '))
            else:
                current_chunk += 1
                chunks.append(sentence.split(' '))
        else:
            chunks.append(sentence.split(' '))

    for chunk_id in range(len(chunks)):
        chunks[chunk_id] = ' '.join(chunks[chunk_id])
    return chunks


def regex_chunking(tokens):
    sent = nltk.pos_tag(tokens)
    regex = "NP: {<DT><JJ>*<NN>}"
    t= nltk.RegexpParser(regex)
    r = t.parse(sent)
    return r
# Simple Regex based NP chunker
from gensim.utils import tokenize
from gensim.parsing.preprocessing import remove_stopwords, strip_punctuation

def tokenize_with_gensim(text):
    # Supprimer la ponctuation
    text = strip_punctuation(text)
    
    # Supprimer les stopwords
    text = remove_stopwords(text)
    
    # Tokenization avec Gensim
    tokens = list(tokenize(text))
    
    return tokens
from textblob import TextBlob

def tokenize_with_textblob(text):
    # Créer un objet TextBlob à partir du texte
    blob = TextBlob(text)
    
    # Tokenization avec TextBlob
    tokens = blob.words
    
    return tokens
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def tokenize_with_nltk(text):
    # Convertir le texte en minuscules
    text = text.lower()
    
    # Tokenization avec NLTK
    tokens = word_tokenize(text)
    
    # Supprimer les stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return tokens


def regex_chunker(tokens):
    sent = nltk.pos_tag(tokens)
    regex = """NP:{<.*>+}               
                    }<VBD | IN>+{"""
    # it will chunk everything first and then chunk VBD and IN
    t = nltk.RegexpParser(regex)
    r = t.parse(sent)
    return r
 #unigram chunker with NP
import spacy

def lemmatize_text1(text):
    nlp = spacy.load('en_core_web_sm')
    lemmatized_text = ""
    doc = nlp(text)
    for token in doc:
        lemmatized_text += token.lemma_ + " "
    return lemmatized_text.strip()



class UnigramChunker(nltk.ChunkParserI):
    def __init__(self, t_sents):
        td = [[(t,c) for w,t,c in nltk.chunk.tree2conlltags(s)]
                      for s in t_sents]
        self.tagger = nltk.UnigramTagger(td)

    def parse(self, s):
        postags = [p for (w,p) in s]
        t_postags = self.tagger.tag(postags)
        ctags = [ctag for (p, ctag) in t_postags]
        conlltags = [(w, p, ctag) for ((w,p),ctag)
                     in zip(s, ctags)]
        return nltk.chunk.conlltags2tree(conlltags)     
       
def hapax_chunker(sentence):
    tokens = sentence.split()
    word_counts = {}
    
    # Compter les occurrences de chaque mot
    for token in tokens:
        if token in word_counts:
            word_counts[token] += 1
        else:
            word_counts[token] = 1
    
    # Extraire les hapax (mots qui n'apparaissent qu'une seule fois)
    hapax_words = [word for word, count in word_counts.items() if count == 1]
    
    # Créer des chunks à partir des hapax
    chunks = []
    current_chunk = []
    for token in tokens:
        if token in hapax_words:
            if current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
        else:
            current_chunk.append(token)
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks
     
nlp = spacy.load("en_core_web_sm")

def ner_extraction(text):
    # Traitement du texte avec Spacy
    doc = nlp(text)
    
    # Extraction des entités nommées
    entities = []
    for ent in doc.ents:
        entities.append((ent.text, ent.label_))
    
    return entities
# Function to Analyse Tokens and Lemma
@st.cache
def text_analyzer(my_text):
	nlp = spacy.load('en_core_web_sm')
	docx = nlp(my_text)
	# tokens = [ token.text for token in docx]
	allData = ['{{"Token":"{}", "Lemma":"{}"}}'.format(token.text, token.lemma_) for token in docx]
	return allData

# Function For Extracting Entities
@st.cache
def entity_analyzer(my_text):
	nlp = spacy.load('en_core_web_sm')
	docx = nlp(my_text)
	tokens = [ token.text for token in docx]
	entities = [(entity.text,entity.label_)for entity in docx.ents]
	allData = ['"Token":{},\n"Entities":{}'.format(tokens,entities)]
	return allData

from nltk.stem import WordNetLemmatizer

def lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in text.split()]
    lemmatized_text = ' '.join(lemmatized_words)
    return lemmatized_text

def read_and_clean_file(file_path):
    # Lire le fichier et obtenir les lignes
    with open(file_path, "r") as file:
        lines = [line.strip() for line in file.readlines()]

    # Créer un DataFrame Pandas avec les lignes comme données
    df = pd.DataFrame({"Phrases": lines})

    # Supprimer les lignes indésirables (par exemple, les 20 premières lignes)
    df = df.drop(range(20))

    # Supprimer les lignes vides ou contenant uniquement des espaces
    df = df[df['Phrases'].str.strip() != '']

    # Réindexer le DataFrame
    df = df.reset_index(drop=True)

    return df   
### Named Entity Recognition

# Using Spacy for Named Entity Recognition.
# Apply spacy on our sentence 
# Named Entity Recognition
def perform_ner(text):
    ner = spacy.load("en_core_web_sm")
    doc = ner(text)

    # Display entities
    for ent in doc.ents:
        st.write(ent.text, ent.label_)

    # Render the entities using displacy
    html = displacy.render(doc, style="ent", page=True)
    st.components.v1.html(html, width=800, height=600, scrolling=True)
    
def ner_extraction(sentence):
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(sentence)
    
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    print(entities)
    
    displacy.render(doc, style='ent', jupyter=True)


# def tokenize(self, text, include_punc=True):
#     tokens = word_tokenize(text)
#     if include_punc:
#         return tokens
#     else:
#         return [
#             word if word.startswith("'") else strip_punc(word, all=False)
#             for word in tokens if strip_punc(word, all=False)
#         ]

# def tokenize(self, text):
#     return sent_tokenize(text)

# Text Summarization 
### TextRank


# TextRank
def analyze_text_Rank(text):
    # Count the number of sentences
    sentences = sent_tokenize(text)
    num_sentences = len(sentences)
    st.write("Number of sentences:", num_sentences)

    # Generate a summary
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = TextRankSummarizer()
    summary = summarizer(parser.document, 2)  # Number of sentences in the summary
    text_summary = " ".join(str(sentence) for sentence in summary)

    st.write("Summary:")
    st.write(text_summary)

# LexRank 
def analyze_text_Lex(text):
    # Count the number of sentences
    sentences = sent_tokenize(text)
    num_sentences = len(sentences)
    st.write("Number of sentences:", num_sentences)

    # Generate a summary using TextRank
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer_textrank = TextRankSummarizer()
    summary_textrank = summarizer_textrank(parser.document, 2)  # Number of sentences in the summary

    text_summary_textrank = " ".join(str(sentence) for sentence in summary_textrank)

    st.write("TextRank Summary:")
    st.write(text_summary_textrank)

    # Generate a summary using LexRank
    summarizer_lexrank = LexRankSummarizer()
    summary_lexrank = summarizer_lexrank(parser.document, 2)  # Number of sentences in the summary

    text_summary_lexrank = " ".join(str(sentence) for sentence in summary_lexrank)

    st.write("LexRank Summary:")
    st.write(text_summary_lexrank)



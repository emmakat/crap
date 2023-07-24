
import joblib
import re
import streamlit as st
import numpy as np
import pandas as pd
import pprint
import warnings
import tempfile
from io import StringIO
from PIL import  Image
from rake_nltk import Rake
import spacy
from spacy import displacy
import spacy_streamlit
from collections import Counter
import en_core_web_sm
from nltk.tokenize import sent_tokenize
from collections import Counter
from PIL import features
# print(features.check('freetype2'))
# Warnings ignore 
warnings.filterwarnings(action='ignore')
st.set_option('deprecation.showfileUploaderEncoding', False)
st.set_option('deprecation.showPyplotGlobalUse', False)
# @st.cache(allow_output_mutation=True)

import text_analysis as nlp
import text_summarize as ts
# from text_analysis import * 

def display_data(data):
    # Display the content of the data
    st.write("Data content:")
    st.write(data)
    
def main():
	option = st.sidebar.selectbox('Navigation', 
				["Home",
				"Data Preprocessing",
				"Tokenization",
				"Lemmatization",
				"Parts of Speech Analysis",
				"Hapax Chunker",
				"Generate Chunks",
				"Keyword Sentiment Analysis",
				"Word Cloud",
				"N-Gram Analysis",
				"Named Entity Recognition",
				"the Tokens and Lemma of text",
				"Summarization"
			])
  


	st.set_option('deprecation.showfileUploaderEncoding', False)
	# Home 

	if option == 'Home':
		st.markdown("<h1 style='text-align: center; color: black; font-family: Verdana, sans-serif;'>Text Analysis Application</h1>", unsafe_allow_html=True)
		display = Image.open('images/download.png')
		st.image(display, width=500)

	# Data Preprocessing
	elif option == "Data Preprocessing":
		st.title("Data Preprocessing")
		file_type = st.selectbox("Select file type", ["Text (txt)", "CSV", "Markdown (md)"])

		# Display a file upload field based on the selected type
		if file_type == "Text (txt)":
			file = st.file_uploader("Upload a text file", type=["txt"])
		elif file_type == "CSV":
			file = st.file_uploader("Upload a CSV file", type=["csv"])
		elif file_type == "Markdown (md)":
			file = st.file_uploader("Upload a Markdown file", type=["md"])

		# Import the data if a file has been selected
		if file is not None:
			data = None

			if file_type == "Text (txt)":
				data = file.read().decode("utf-8")
				content_placeholder = st.empty()  # Placeholder for displaying content dynamically
				if st.button("Display Data"):
					content_placeholder.write("Content of the text file:")
					display_data(data)
				if st.button("Clean Data"):
					cleaned_data = nlp.clean_text(data)  # Perform text cleaning (replace with appropriate cleaning function)
					content_placeholder.write("Cleaned data:")
					display_data(cleaned_data)
			
			elif file_type == "CSV":
				data = pd.read_csv(file)
				content_placeholder = st.empty()  # Placeholder for displaying content dynamically
				if st.button("Display Data"):
					content_placeholder.write("Data imported from the CSV file:")
					display_data(data)
				if st.button("Clean Data"):
					cleaned_data = nlp.clean_text(data)  # Perform text cleaning (replace with appropriate cleaning function)
					content_placeholder.write("Cleaned data:")
					display_data(cleaned_data)
			
			elif file_type == "Markdown (md)":
				data = file.read().decode("utf-8")
				content_placeholder = st.empty()				 
				if st.button("Display Data"):
					st.write("Data imported from the MD file:")
					display_data(data)
				if st.button("Clean Data"):		
					cleaned_data = nlp.clean_text(data)  # Perform text cleaning (replace with appropriate cleaning function)
					content_placeholder.write("Cleaned data:")
					display_data(cleaned_data)
		


	elif option == "Tokenization":
		st.title("Tokenization")
		data_input = st.radio("Select data input option:", ("Upload File", "Enter Text"))

		if data_input == "Upload File":
			uploaded_file = st.file_uploader("Upload a file", type=["txt"])
			if uploaded_file is not None:
				text = uploaded_file.read().decode("utf-8")
				st.write("Uploaded file content:")
				st.write(text)
			else:
				text = ""  # Set empty text if no file is uploaded
		else:
			text = st.text_area("Enter your text here")

		tokenization_model = st.selectbox("Select a tokenization model:", ("Gensim", "TextBlob", "NLTK"))

		show_results = st.button("Show Results")
		show_all_results = st.button("Show All Results")

		if show_results:
			if tokenization_model == "Gensim":
				st.subheader("Tokens generated with Gensim:")
				gensim_tokens = nlp.tokenize_with_gensim(text)
				st.write(gensim_tokens)
			elif tokenization_model == "TextBlob":
				st.subheader("Tokens generated with TextBlob:")
				textblob_tokens = nlp.tokenize_with_textblob(text)
				st.write(textblob_tokens)
			elif tokenization_model == "NLTK":
				st.subheader("Tokens generated with NLTK:")
				nltk_tokens = nlp.tokenize_with_nltk(text)
				st.write(nltk_tokens)

		if show_all_results:
			gensim_tokens = nlp.tokenize_with_gensim(text)
			textblob_tokens = nlp.tokenize_with_textblob(text)
			nltk_tokens = nlp.tokenize_with_nltk(text)

			# Vérifier la longueur des listes
			if len(gensim_tokens) == len(textblob_tokens) == len(nltk_tokens):
				# Créer le dictionnaire data
				data = {
					"Gensim Tokens": gensim_tokens,
					"TextBlob Tokens": textblob_tokens,
					"NLTK Tokens": nltk_tokens
				}

				# Créer le DataFrame
				df = pd.DataFrame(data)
				st.dataframe(df)
			else:
				st.error("Error: The lists have different lengths.")


	elif option == "Word Cloud":
		# Word Cloud Feature
		st.header("Generate Word Cloud")
		st.subheader("Generate a word cloud from text containing the most popular words in the text.")

		# Choose between text input or file upload
		data_option = st.radio("Choose data option", ("Enter text", "Upload a file"))

		if data_option == "Enter text":
			text = st.text_area("Type Something", height=400)
		else:
			file = st.file_uploader("Upload a file", type=["txt"])
			if file is not None:
				text = file.read().decode("utf-8")
		
		# Upload mask image
		mask = st.file_uploader('Use Image Mask', type=['jpg'])

		# Add a button feature
		if st.button("Generate Wordcloud"):
			# Generate word cloud
			nlp.create_wordcloud(text, mask)
			st.pyplot()


	# N-Gram Analysis Option 
	elif option == "N-Gram Analysis":
		st.header("N-Gram Analysis")
		st.subheader("This section displays the most commonly occurring N-Grams in your Data")

		# Ask for text or text file
		st.header('Enter text or upload a text file')
		input_option = st.radio("Select Input Option", ("Enter Text", "Upload Text File"))

		if input_option == "Enter Text":
			text = st.text_area('Type Something', height=400)
		else:
			file = st.file_uploader("Upload a text file", type=["txt"])
			if file is not None:
				text = file.read().decode("utf-8")
			else:
				text = ""

		# Parameters
		n = st.sidebar.slider("N for the N-gram", min_value=1, max_value=8, step=1, value=2)
		topk = st.sidebar.slider("Top k most common phrases", min_value=10, max_value=50, step=5, value=10)

		# Add a button
		if st.button("Generate N-Gram Plot"):
			# Plot the ngrams
			nlp.plot_ngrams(text, n=n, topk=topk)
			st.pyplot()

	# POS Tagging Option
	elif option == "Parts of Speech Analysis":
		st.header("Enter the statement that you want to analyze")
		
		# Choose between file import or text input
		data_option = st.radio("Choose data option", ("Upload a file", "Enter text"))
		
		if data_option == "Upload a file":
			file = st.file_uploader("Upload a file")
			
			# Import the data if a file has been selected
			if file is not None:
				data = file.read().decode("utf-8")
				text_input = st.text_area("Content of the file:", value=data)
		else:
			text_input = st.text_input("Enter sentence", '')
		
		if st.button("Show POS Tags"):
			tags = nlp.pos_tagger(text_input)
			st.markdown("The POS Tags for this sentence are: ")
			st.markdown(tags, unsafe_allow_html=True)
			
			st.markdown("### Penn-Treebank Tagset")
			st.markdown("The tags can be referenced from here:")
			
			# Show image
			display_pos = Image.open('images/Penn_Treebank.png')
			display_pos = np.array(display_pos)
			st.image(display_pos)

    	# Tokenization

	elif option == 'the Tokens and Lemma of text':
		st.header("Enter Text")
		# Choose between file import or text input
		data_option = st.radio("Choose data option", ("Upload a file", "Enter text"))

		if data_option == "Upload a file":
			file = st.file_uploader("Upload a file")

			# Import the data if a file has been selected
			if file is not None:
				data = file.read().decode("utf-8")
				text_input = st.text_area("Content of the file:", value=data)
		else:
			text_input = st.text_area("Enter sentence")

		if st.button("Analyze"):
			nlp_result = nlp.text_analyzer(text_input)
			st.json(nlp_result)

 
		
	# Named Entity Recognition	
	elif option == "Named Entity Recognition":
		st.header("Named Entity Recognition")

		# Choose between file import or text input
		data_option = st.radio("Choose data option", ("Upload a file", "Enter text"))

		if data_option == "Upload a file":
			file = st.file_uploader("Upload a file")

			# Import the data if a file has been selected
			if file is not None:
				data = file.read().decode("utf-8")
				text_input = st.text_area("Content of the file:", value=data)
			else:
				text_input = None
		else:
			text_input = st.text_area("Enter sentence")

		if text_input is not None and text_input.strip():
			nlp.perform_ner(text_input.strip())
		else:
			st.write("Please enter a sentence or upload a file")


	# SUMMARAZATION
	elif option == "Summarization":
		st.title("Text Summarization")
		text_input = st.text_area("Enter your text")
		option = st.selectbox("Select Summarization Method", ("TextRank", "LexRank"))

		if st.button("Generate Summary"):
			if option == "TextRank":
				nlp.analyze_text_Rank(text_input)
			elif option == "LexRank":
				nlp.analyze_text_Lex(text_input)

	# Generate Chunks
	elif option =="Generate Chunks": 
		text_input = st.text_area("Enter your text")
		
		if st.button("Generate Chunks"):
			chunks = nlp.generate_chunks(text_input)
			
			if chunks:
				st.header("Generated Chunks:")
				for chunk in chunks:
					st.write(chunk)
	# Keyword Sentiment Analysis
	elif option == "Keyword Sentiment Analysis":
		st.header("Sentiment Analysis Tool")
		st.subheader("Enter the statement that you want to analyze")
		
		# Choose between file import or text input
		data_option = st.radio("Choose data option", ("Upload a file", "Enter text"))
		
		if data_option == "Upload a file":
			file = st.file_uploader("Upload a file")
			
			# Import the data if a file has been selected
			if file is not None:
				data = file.read().decode("utf-8")
				text_input = st.text_area("Content of the file:", value=data, height=50)
		else:
			text_input = st.text_area("Enter sentence", height=50)
		
		# Model Selection 
		model_select = st.selectbox("Model Selection", ["Naive Bayes", "SVC", "Logistic Regression"])
		
		if st.button("Predict"):
			# Load the model
			if model_select == "SVC":
				sentiment_model = joblib.load('Models/SVC_sentiment_model.sav')
			elif model_select == "Logistic Regression":
				sentiment_model = joblib.load('Models/LR_sentiment_model.sav')
			elif model_select == "Naive Bayes":
				sentiment_model = joblib.load('Models/NB_sentiment_model.sav')
			
			# Vectorize the inputs
			vectorizer = joblib.load('Models/tfidf_vectorizer_sentiment_model.sav')
			vec_inputs = vectorizer.transform([text_input])
			
			# Keyword extraction
			r = Rake(language='english')
			r.extract_keywords_from_text(text_input)
			
			# Get the important phrases
			phrases = r.get_ranked_phrases()
			
			# Remove repeated keywords
			unique_phrases = list(dict(Counter(phrases)).keys())
			
			# Make the prediction
			if sentiment_model.predict(vec_inputs):
				st.write("This statement is **Positive**")
			else:
				st.write("This statement is **Negative**")
			
			# Display the unique keywords
			st.write("These are the **keywords** causing the above sentiment:")
			for i, p in enumerate(unique_phrases):
				st.write(i+1, p)

	# Hapax Chunker
	elif option == "Hapax Chunker":
		st.title("Hapax Chunker")
		
		sentence = st.text_input("Enter a sentence:")
		
		if st.button("Apply"):
			entities = nlp.hapax_chunker(sentence)
			st.subheader("Named Entities:")
			for entity in entities:
				st.write(entity)

	# Lemmatization
	elif option == "Lemmatization":
		st.title("Lemmatization")

		# Choose between file import or text input
		data_option = st.radio("Choose data option", ("Upload a file", "Enter text"))

		if data_option == "Upload a file":
			file = st.file_uploader("Upload a file")

			# Import the data if a file has been selected
			if file is not None:
				data = file.read().decode("utf-8")

				if st.button("Apply"):
					lemmatized_text = nlp.lemmatize_text1(data)
					st.subheader("Lemmatized Text:")
					st.write(lemmatized_text)
		else:
			text = st.text_area("Enter text:")

			if st.button("Apply"):
				lemmatized_text = nlp.lemmatize_text1(text)
				st.subheader("Lemmatized Text:")
				st.write(lemmatized_text)





			
if __name__ == "__main__":
    main()

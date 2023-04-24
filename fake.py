#!  /usr/bin/python
import pandas as pd
import numpy as np
import os
import lightgbm as lgb
import argparse

from sklearn.metrics import accuracy_score, recall_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
import sys as sys
from sklearn.feature_extraction.text import TfidfVectorizer
import gensim.downloader as api
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pickle

additional_stopwords = []
def basic_preprocessing(text):
	"""
	A simple function to clean up the data. All the words that
	are not designated as a stop word is then lemmatized after
	encoding and basic regex parsing are performed.
	"""
	# Convert to lowercase
	text = text.lower()

	# Remove punctuation
	text = text.translate(str.maketrans('', '', string.punctuation))
	# Tokenize the text
	tokens = text.split()
	# Remove stopwords
	stop_words = set(stopwords.words('english') + additional_stopwords)
	tokens = [token for token in tokens if token not in stop_words]
	# Lemmatize words
	lemmatizer = WordNetLemmatizer()
	tokens = [lemmatizer.lemmatize(token) for token in tokens]

	# Join the tokens back into a string
	return ' '.join(tokens)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("-input_path", "-i",
						help="Folder with data files",
						default='./data/archive')



	args = parser.parse_args()
	CSV_PATH = args.input_path

	### Read training files #####
	fake = pd.read_csv(os.path.join(CSV_PATH, 'Fake.csv'))
	fake['label'] = 0
	real = pd.read_csv(os.path.join(CSV_PATH, 'True.csv'))
	real['label'] = 1
	# Concat to one dataframe
	dataset  = pd.concat([real, fake],  ignore_index=True)
	# Not any particular reason, I just prefer to have the two classes shuffled
	dataset = dataset.sample(frac=1).reset_index(drop=True)

	# Data Preparation
	dataset.text  = dataset.text.apply(lambda row: basic_preprocessing(row))

	# Splitting Data
	X_train, X_test, y_train, y_test = train_test_split(dataset.text, dataset.label, test_size=0.2, random_state=42)


	# Tokenizing and embedding
	w2v_model = api.load('word2vec-google-news-300')
	embedding_size = w2v_model.vector_size

	# Feature Extraction
	vectorizer = TfidfVectorizer(analyzer=lambda x: x.split(), tokenizer=lambda x: x.split(),
								 preprocessor=lambda x: x, token_pattern=None)
	vectorizer.fit(X_train)
	X_train_tfidf = vectorizer.transform(X_train)
	# X_train_emb = np.array([np.mean(
	# 	[w2v_model[w] * vectorizer.idf_[vectorizer.vocabulary_[w]] for w in text.split() if
	# 	 w in w2v_model and w in vectorizer.vocabulary_] or [np.zeros(embedding_size)], axis=0) for
	# 						text in X_train])

	X_test_tfidf = vectorizer.transform(X_test)
	# X_test_emb = np.array([np.mean(
	# 	[w2v_model[w] * vectorizer.idf_[vectorizer.vocabulary_[w]] for w in text.split() if
	# 	 w in w2v_model and w in vectorizer.vocabulary_] or [np.zeros(embedding_size)], axis=0) for
	# 					   text in X_test])

	# Feature Extraction
	# X_train_emb = np.array([np.mean(
	# 	[w2v_model[w] for w in text.split() if w in w2v_model] or [np.zeros(embedding_size)],
	# 	axis=0) for text in X_train_spm])
	# X_test_emb = np.array([np.mean(
	# 	[w2v_model[w] for w in text.split() if w in w2v_model] or [np.zeros(embedding_size)],
	# 	axis=0) for text in X_test_spm])


	# Training Classifier
	clf = lgb.LGBMClassifier(num_leaves=64, n_estimators=300, max_depth=9)
	# CV_rfc = GridSearchCV(clf, param_grid={'max_depth': [2, 3]}, cv=5)
	clf.fit(X_train_tfidf, y_train)
	# save the model to disk
	filename = 'finalized_model.sav'
	pickle.dump(clf, open(filename, 'wb'))

	# Evaluation
	y_pred = clf.predict(X_test_tfidf)
	accuracy = accuracy_score(y_test, y_pred)
	print("Accuracy:", accuracy)

	scores = cross_val_score(clf, X_test_tfidf, y_test, cv=5)
	print("==============================================")
	for score in scores:
		print("Accuracy:", scores)
	print("Mean Accuracy:", accuracy)
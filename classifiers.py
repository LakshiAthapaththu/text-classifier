import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import random
from keras.layers import Input, Embedding, Dot, Reshape, Dense
from keras.models import Model
from sklearn.linear_model import LogisticRegression
from numpy import array, matrix
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
import pickle


def readData():
	data = pd.read_csv("TrainingCorpus.csv")	
	test = pd.read_csv("testset.csv")
	return data,test


def getCorpus(corpus):
	word_array = []
	for sentence in corpus:
		word_array.extend(str(sentence).split())
		word_array = (set(word_array))
		word_array= list(word_array)

	return word_array

def generateWordBags(corpus,word_array):
	sentence_bags = []
	for sentence in corpus:
		bag = [0]*len(word_array)
		for word in (str(sentence).split()):
			if(word in word_array):
				bag[(word_array.index(word))] = 1
		sentence_bags.append(bag)	
	return sentence_bags


def trainClassifiers(word_bags,label_bags,X_train_tfidf):
	LRclassifier = LogisticRegression()
	LRclassifier.fit(word_bags,label_bags)
	NBclassifier = MultinomialNB().fit(X_train_tfidf, label_bags)
	SVMclassifier = SGDClassifier().fit(X_train_tfidf, label_bags)
	
	return LRclassifier,NBclassifier,SVMclassifier
	

def scoreClassifier(word_bags_test,label_bags_test,classifier):
	score = classifier.score(word_bags_test,label_bags_test)
	return score




def gettfidf(sentence_list):
	X_train_counts = csr_matrix(sentence_list)
	tfidf_transformer = TfidfTransformer()
	X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
	return X_train_tfidf

def getWordbag(word_array,sentence):
	bag = [0]*len(word_array)
	for word in (str(sentence).split()):
		if(word in word_array):
			bag[(word_array.index(word))] = 1
	return bag


def predict(sentence,classifier):
	prediction = classifier.predict(sentence)
	return prediction
	
	

training,test = readData()

training_corpus = list(training.utter)
training_labels = list(training.intent)

test_corpus = list(test.utter)
test_labels = list(test.intent)


word_array = (getCorpus(training_corpus))
training_sentence_bag = (generateWordBags(training_corpus,word_array))
test_sentence_bag = (generateWordBags(test_corpus,word_array))



X_train_tfidf = gettfidf(training_sentence_bag)
X_test_tfidf = gettfidf(test_sentence_bag)

LRclassifier, NBclassirier,SVMclassifier = trainClassifiers(training_sentence_bag,training_labels,X_train_tfidf)
scoreLR = scoreClassifier(test_sentence_bag,test_labels,LRclassifier)
scoreNB = scoreClassifier(X_test_tfidf,test_labels,NBclassirier)
scoreSVM = scoreClassifier(X_test_tfidf,test_labels,SVMclassifier)

print(scoreLR)
print(scoreNB)
print(scoreSVM)

filenameSVM = 'finalized_model_SVM.sav'
filenameNB = 'finalized_model_NB.sav'
filenameLR = 'finalized_model_LR.sav'

pickle.dump(NBclassirier, open(filenameNB, 'wb'))
pickle.dump(SVMclassifier, open(filenameSVM, 'wb'))
pickle.dump(LRclassifier, open(filenameLR, 'wb'))

with open('vocab.data', 'wb') as filehandle:
    # store the data as binary data stream
    pickle.dump(word_array, filehandle)














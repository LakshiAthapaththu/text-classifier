import pickle
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfTransformer
import pandas as pd
from rasa_nlu.model import Interpreter
import json

def predict(sentence,classifier):
	prediction = classifier.predict(sentence)
	return prediction


def getWordbag(word_array,sentence):
	bag = [0]*len(word_array)
	for word in (str(sentence).split()):
		if(word in word_array):
			bag[(word_array.index(word))] = 1
	return bag


def gettfidf(sentence_list):
	X_train_counts = csr_matrix(sentence_list)
	tfidf_transformer = TfidfTransformer()
	X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
	return X_train_tfidf

def generateWordBags(corpus,word_array):
	sentence_bags = []
	for sentence in corpus:
		bag = [0]*len(word_array)
		for word in (str(sentence).split()):
			if(word in word_array):
				bag[(word_array.index(word))] = 1
		sentence_bags.append(bag)	
	return sentence_bags

def loadmodels():
		
	filenameSVM = 'finalized_model_SVM.sav'
	filenameNB = 'finalized_model_NB.sav'
	filenameLR = 'finalized_model_LR.sav'

	loaded_model_SVM = pickle.load(open(filenameSVM, 'rb'))
	loaded_model_NB = pickle.load(open(filenameNB, 'rb'))
	loaded_model_LR = pickle.load(open(filenameLR, 'rb'))

	with open('vocab.data', 'rb') as filehandle:
	# read the data as binary data stream
		word_array_read = pickle.load(filehandle)
	return loaded_model_SVM,loaded_model_NB,loaded_model_LR,word_array_read

def predictEnsemble(loaded_model_SVM,loaded_model_NB,loaded_model_LR,sentence_tfidf,bag,sentence,interpreter):
	predSVM = predict(sentence_tfidf,loaded_model_SVM)[0]
	predNB = predict(sentence_tfidf,loaded_model_NB)[0]
	predLR = predict([bag],loaded_model_LR)[0]
	word_embedding_result = interpreter.parse(sentence)
	intent_from_embedding = word_embedding_result['intent']['name']
	l = [predSVM,predNB,predLR,intent_from_embedding]
	most_voted = max(set(l), key = l.count)
	return most_voted



interpreter = Interpreter.load("./models/current/nlu")
file_read = open('test_sentences','r')

test = pd.read_csv("testset.csv")
test_corpus = list(test.utter)
test_labels = list(test.intent)



#sentence = "ආයෙත් උත්සහ කරන්න ඕන නැහැ"
loaded_model_SVM,loaded_model_NB,loaded_model_LR,word_array_read = loadmodels()

for line in file_read:
	sentence = line
	sentence_bag = getWordbag(word_array_read,sentence)
	sentence_tfidf = gettfidf([sentence_bag])
	most_voted = predictEnsemble(loaded_model_SVM,loaded_model_NB,loaded_model_LR,sentence_tfidf,sentence_bag,sentence,interpreter)
	print("intent",most_voted)






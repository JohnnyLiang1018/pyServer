import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from scipy import sparse
import joblib

def process(input):
	data = {
		"goal": int(input.get("goal")),
		"name": input.get("name"),
		"blurb": input.get("blurb"),
		"Length_of_kick": int(input.get("Length_of_kick"))
	}
	preProcessingData = preprocess(data)
	return {"data": input, "prediction": predict(preProcessingData)}

def preprocess(data):
	# 1
	# data is dictionary
	row = [data]
	df = pd.DataFrame(row) 
	df['blurb'] = [str(entry).lower() for entry in df['blurb']]
	df['blurb'] = [word_tokenize(entry) for entry in df['blurb']]
	tag_map = defaultdict(lambda : wn.NOUN)
	tag_map['J'] = wn.ADJ
	tag_map['V'] = wn.VERB
	tag_map['R'] = wn.ADV
	for index,entry in enumerate(df['blurb']):
	    # Declaring Empty List to store the words that follow the rules for this step
	    Final_words = []
	    # Initializing WordNetLemmatizer()
	    word_Lemmatized = WordNetLemmatizer()
	    # pos_tag function below will provide the 'tag' i.e if the word is Noun(N) or Verb(V) or something else.
	    for word, tag in pos_tag(entry):
	        # Below condition is to check for Stop words and consider only alphabets
	        if word not in stopwords.words('english') and word.isalpha():
	            word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
	            Final_words.append(word_Final)
	    # The final processed set of words for each iteration will be stored in 'text_final'
	    df.loc[index,'word_token'] = str(Final_words)
	    print('progress: ',index)
	df['word_token'] = df['word_token'].astype(str)
	# 3
	df2 = df[['goal','Length_of_kick']]

	# 4
	# save vector
	filename = 'word_vect.sav'
	count_vect = joblib.load(filename)
	x_count = count_vect.transform(df['word_token'])
	# 5
	df3 = pd.DataFrame(x_count)
	# 6
	df4 = pd.concat([df2,df3],axis=1)
	# 7
	x_1 = df4[['goal']].values
	x_2 = df4[['Length_of_kick']].values
	# 8
	X_input = sparse.hstack((x_count,x_1,x_2))
	# 9
	X_input.data = np.nan_to_num(X_input.data)
	# 10
	X_input.eliminate_zeros()
	return X_input

def predict(input):
	filename = 'mlp_model.sav'
	loaded_model = joblib.load(filename)
	predicted = loaded_model.predict_proba(input)
	return predicted
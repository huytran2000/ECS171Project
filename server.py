
from flask import Flask, request, render_template, jsonify
import os

from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

app = Flask(__name__)

def del_index_list(list_object, indices):
    indices = sorted(indices, reverse=True)
    for i in indices:
        if i < len(list_object):
            #print("before:", list_object)
            list_object.pop(i)
            #print("after:", list_object)

#classifier = 0
#df = 0
#df_shape = 0#comparable with 0
#to_drop_returned = 0

def init_train_data():#failed optimization
	colNames = list(range(1024))
	colNames.append("label")
	df = pd.read_csv('./qsar_oral_toxicity.csv', sep=';',names=colNames)
	df = pd.get_dummies(df)
	df.drop("label_negative",1,inplace=True)
	df = df.rename({"label_positive":"label"},axis=1)

	# Create correlation matrix
	corr_matrix = df.corr().abs()
	# Select upper triangle of correlation matrix (not including diagonal)
	upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
	# Find features with correlation greater than 0.95
	to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
	# Drop features 
	df.drop(to_drop, axis=1, inplace=True)

	df_shape = df.shape
	print("here")
	return to_drop

def model(in_arr):
	threshold = 0.28
	#optimize perf

	colNames = list(range(1024))
	colNames.append("label")
	df = pd.read_csv('./qsar_oral_toxicity.csv', sep=';',names=colNames)
	df = pd.get_dummies(df)
	df.drop("label_negative",1,inplace=True)
	df = df.rename({"label_positive":"label"},axis=1)
	#print(3)
	#print(type(df))
	# Create correlation matrix
	corr_matrix = df.corr().abs()
	# Select upper triangle of correlation matrix (not including diagonal)
	upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
	# Find features with correlation greater than 0.95
	to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
	# Drop features 
	df.drop(to_drop, axis=1, inplace=True)

	del_index_list(in_arr, to_drop)

	X_train = df.drop(["label"],1)
	y_train = df["label"]
	X_train.reset_index(drop=True, inplace=True)

	val = pd.Series(in_arr)
	classifier = RandomForestClassifier(criterion="entropy", max_depth=63, max_features="auto", n_estimators=603)
	classifier.fit(X_train, y_train)
	res = classifier.predict_proba([val])[:,1][0]
	if res >= threshold:
	    label = "VERY TOXIC"
	else:
	    label = "NOT VERY TOXIC"
	return label

#Upper just capitalizes the message
def checkValid(encoding):
    invalidAttributes = []
    try:
        int(encoding)
        print("type of input is", type(encoding))
        if len(encoding) != 1024:
            invalidAttributes.append('encoding')
            print("in")
        for i in [int(x) for x in encoding]:
        	if i==0 or i==1:
        		continue
        	invalidAttributes.append('encoding')
        	break
        print(encoding)
        print(encoding)
    except:
        invalidAttributes.append('encoding')
    print('|',encoding,'|')
    print(type(encoding))
    print(invalidAttributes)
    return invalidAttributes


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/getModelOutput', methods=['GET','POST'])

def my_form_post():
    data_input = request.form['code']
    print("str is", data_input)
    print("str is", request.form)

    invalidFeat = checkValid(data_input)

    if invalidFeat:
        #some value is incorrect
        print("Something was incorrect")
        result = {
            "output1": "Invalid input detected",
        }

    else:

        data_array = [int(x) for x in data_input]
        output = model(data_array)
        print("All well")
        #print(data_in)
        result = {
            "output": output
        }
    result = {str(key): value for key, value in result.items()}
    return jsonify(result=result)

if __name__ == '__main__':
    app.run(debug=False)

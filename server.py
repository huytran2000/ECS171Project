
from flask import Flask, request, render_template, jsonify
import os

from sklearn.ensemble import RandomForestClassifier
import pandas as pd

app = Flask(__name__)

def model(in_arr):
    colNames = list(range(1024))
    colNames.append("label")
    df = pd.read_csv('./qsar_oral_toxicity.csv', sep=';',names=colNames)
    df = pd.get_dummies(df)
    df.drop("label_negative",1,inplace=True)
    df = df.rename({"label_positive":"label"},axis=1)
    X_train = df.drop(["label"],1)
    y_train = df["label"]
    X_train.reset_index(drop=True, inplace=True)

    val = pd.Series(in_arr)
    classifier = RandomForestClassifier(n_estimators=500, max_depth=70)
    classifier.fit(X_train, y_train)
    res = classifier.predict([val])[0]
    if res == 1:
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

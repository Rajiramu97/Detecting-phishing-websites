# -*- coding: utf-8 -*-

#importing libraries
from sklearn.externals import joblib
import warnings
import inputScript

warnings.filterwarnings('ignore')
#load the pickle file
classifier = joblib.load('final_models/svm_final.pkl')

#input url
print("enter url")
url = input()

#checking and predicting
checkprediction = inputScript.main(url)
prediction = classifier.predict(checkprediction)
print(prediction)

if prediction==-1:
	print("Not a phishing website")
else:
	print("Phishing website")

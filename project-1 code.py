import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
#loading the csv data to a pandas DataFrame
heart_data = pd.read_csv("C:\Users\Bhavana\Documents\SYLLABUS\codes\Major project\cardio_train.csv")
#print first 5 rows of the dataset
heart_data.head()
#print last 5 rows of the dataset
heart_data.tail()
#noof rows and colums in the dataset
heart_data.shape
#getting some info about the data
heart_data.info()
#checking for missing values
heart_data.isnull().sum()
#statistical measures about the data
heart_data.describe()
# checking the distribution of Target variable
heart_data['cardio'].value_counts()
X = heart_data.drop(columns='cardio',axis=1)
Y = heart_data['cardio']
print(X)
print(Y)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
print(X.shape, X_train.shape, X_test.shape)
model = LogisticRegression()
# training the LogisticRegression model with Training data
model.fit(X_train, Y_train)
# accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy on training data : ',training_data_accuracy)
# accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy on test data : ',test_data_accuracy)
input_data = ('heart_data')

# change the input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the numpy array as we are predicting for only one instance
input_data_reshape = input_data_as_numpy_array.reshape(1, -1)

prediction = model.predict(input_data_reshape)
print(prediction)

if (prediction[0]== 0):
    print('The person does not have a Cardiovascular Disease')
else:
        print('The person has Cardiovascular Disease')
"""
File: titanic_level2.py
----------------------------------
This file builds a machine learning algorithm by pandas and sklearn libraries.
We use pandas to read in dataset, store data into a DataFrame,
standardize the data by sklearn, and finally train the model and
test it on kaggle website. Hyper-parameters tuning are not required due to its
high level of abstraction, which makes it easier to use but less flexible.
"""

import math
import pandas as pd
from sklearn import preprocessing, linear_model

TRAIN_FILE = 'titanic_data/train.csv'
TEST_FILE = 'titanic_data/test.csv'


def data_preprocess(filename, mode='Train', training_data=None):
	"""
	:param filename: str, the filename to be read into pandas
	:param mode: str, indicating the mode we are using (either Train or Test)
	:param training_data: DataFrame, a 2D data structure that looks like an excel worksheet
						  (You will only use this when mode == 'Test')
	:return: Tuple(data, labels), if the mode is 'Train'; or return data, if the mode is 'Test'
	"""
	# Drop excluded features.
	data = pd.read_csv(filename).drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

	# Modify data to fit training requirements.
	data.Sex.replace(['male', 'female'], [1, 0], inplace=True)
	data.Embarked.replace(['S', 'C', 'Q'], [0, 1, 2], inplace=True)

	if mode == 'Train':
		# Drop empty values.
		data = data.dropna()

		# Separate 'Survived' as labels.
		labels = data.pop('Survived')

		return data, labels

	elif mode == 'Test':
		# Calculate mean for test set (Age and Fare)
		age_mean = round(training_data.Age.mean(), 3)
		fare_mean = round(training_data.Fare.mean(), 3)

		# Fill empty values.
		data.Age.fillna(age_mean, inplace=True)
		data.Fare.fillna(fare_mean, inplace=True)

		return data


def one_hot_encoding(data, feature):
	"""
	:param data: DataFrame, key is the column name, value is its data
	:param feature: str, the column name of interest
	:return data: DataFrame, remove the feature column and add its one-hot encoding features
	"""
	# One-hot the feature.
	feature_dummies = pd.get_dummies(data[feature], prefix=feature)
	if feature == 'Pclass':
		feature_dummies.rename(columns={'Pclass_1': 'Pclass_0', 'Pclass_2': 'Pclass_1', 'Pclass_3': 'Pclass_2'},
							   inplace=True)

	# Remove the original feature.
	data.pop(feature)

	# Concatenate the original data with the one-hot encoded feature.
	data = pd.concat([data, feature_dummies], axis=1)
	return data


def standardization(data, mode='Train'):
	"""
	:param data: DataFrame, key is the column name, value is its data
	:param mode: str, indicating the mode we are using (either Train or Test)
	:return data: DataFrame, standardized features
	"""
	scaler = preprocessing.StandardScaler()

	if mode == 'Train':
		data = scaler.fit_transform(data)

	else:
		data = scaler.transform(data)

	return data


def main():
	"""
	You should call data_preprocess(), one_hot_encoding(), and
	standardization() on your training data. You should see ~80% accuracy on degree1;
	~83% on degree2; ~87% on degree3.
	Please write down the accuracy for degree1, 2, and 3 respectively below
	(rounding accuracies to 8 decimal places)
	TODO: real accuracy on degree1 -> 80.19662921%
	TODO: real accuracy on degree2 -> 83.70786517%
	TODO: real accuracy on degree3 -> 87.64044944%
	"""

	# Pre-process data.
	preprocessed_data, Y = data_preprocess(TRAIN_FILE)

	# One-hot encode data.
	temp = one_hot_encoding(preprocessed_data, 'Sex')
	temp = one_hot_encoding(temp, 'Pclass')
	temp = one_hot_encoding(temp, 'Embarked')

	# Standardize data.
	scaler = preprocessing.StandardScaler()
	X_train = scaler.fit_transform(temp)

	"""
	Degree 1 Polynomial Model
	"""
	h = linear_model.LogisticRegression(max_iter=10000)
	classifier = h.fit(X_train, Y)
	training_acc = classifier.score(X_train, Y)

	# Print degree 1 accuracy.
	print(f'Degree 1 Training Acc: {round(training_acc*100, 8)}%')

	"""
	Degree 2 Polynomial Model
	"""
	poly_phi_2 = preprocessing.PolynomialFeatures(degree=2)
	X_train_poly_2 = poly_phi_2.fit_transform(X_train)

	classifier_poly_2 = h.fit(X_train_poly_2, Y)
	training_acc_2 = classifier_poly_2.score(X_train_poly_2, Y)

	print(f'Degree 2 Training Acc: {round(training_acc_2*100, 8)}%')

	"""
	Degree 3 Polynomial Model
	"""
	poly_phi_3 = preprocessing.PolynomialFeatures(degree=3)
	X_train_poly_3 = poly_phi_3.fit_transform(X_train)

	classifier_poly_3 = h.fit(X_train_poly_3, Y)
	training_acc_3 = classifier_poly_3.score(X_train_poly_3, Y)

	print(f'Degree 3 Training Acc: {round(training_acc_3*100, 8)}%')


if __name__ == '__main__':
	main()

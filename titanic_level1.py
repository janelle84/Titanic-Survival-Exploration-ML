"""
File: titanic_level1.py
Name: Janelle
----------------------------------
This file builds a machine learning algorithm from scratch 
by Python. We'll be using 'with open' to read in dataset,
store data into a Python dict, and finally train the model and 
test it on kaggle website. This model is the most flexible among all
levels. You should do hyper-parameter tuning to find the best model.
"""

import math
from util import *
from collections import defaultdict
TRAIN_FILE = 'titanic_data/train.csv'
TEST_FILE = 'titanic_data/test.csv'


def data_preprocess(filename: str, data: dict, mode='Train', training_data=None):
	"""
	:param filename: str, the filename to be processed
	:param data: an empty Python dictionary
	:param mode: str, indicating if it is training mode or testing mode
	:param training_data: dict[str: list], key is the column name, value is its data
						  (You will only use this when mode == 'Test')
	:return data: dict[str: list], key is the column name, value is its data
	"""

	if mode == 'Train':
		start = 2
		headers = ['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']

		# Read in data.
		with open(filename, 'r') as f:
			header = True
			for line in f:
				data_lst = line.strip().split(',')

				# Handle headers.
				if header:
					for h in headers:
						data[h] = []
					header = False

				# Drop rows with missing values in 'Age' or 'Embarked'.
				elif data_lst[6] == '' or data_lst[12] == '':
					continue

				# Append data with comprehensive features to the dictionary.
				else:
					for i in range(len(data_lst)):
						cur = data_lst[i]
						if i == 1:
							data['Survived'].append(int(cur))

						elif i == start:
							data['Pclass'].append(int(cur))

						elif i == start + 3:
							data['Sex'].append(1) if cur == 'male' else data['Sex'].append(0)

						elif i == start + 4:
							data['Age'].append(float(cur))

						elif i == start + 5:
							data['SibSp'].append(int(cur))

						elif i == start + 6:
							data['Parch'].append(int(cur))

						elif i == start + 8:
							data['Fare'].append(float(cur))

						elif i == start + 10:
							if cur == 'S':
								data['Embarked'].append(0)
							elif cur == 'C':
								data['Embarked'].append(1)
							else:
								data['Embarked'].append(2)

	else:
		headers = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
		start = 1

		# Read in data.
		with open(filename, 'r') as f:
			header = True
			for line in f:
				data_lst = line.strip().split(',')

				# Handle headers.
				if header:
					for h in headers:
						data[h] = []
					header = False

				# Append data with updated features to the dictionary.
				else:
					for i in range(len(data_lst)):
						cur = data_lst[i]
						if i == start:
							data['Pclass'].append(int(cur))

						elif i == start + 3:
							data['Sex'].append(1) if cur == 'male' else data['Sex'].append(0)

						elif i == start + 4:
							data['Age'].append(float(cur)) if cur else data['Age'].append(
								round(sum(training_data['Age']) / len(training_data['Age']), 3))

						elif i == start + 5:
							data['SibSp'].append(int(cur))

						elif i == start + 6:
							data['Parch'].append(int(cur))

						elif i == start + 8:
							data['Fare'].append(float(cur)) if cur else data['Fare'].append(
								round(sum(training_data['Fare']) / len(training_data['Fare']), 3))

						elif i == start + 10:
							if cur == 'S':
								data['Embarked'].append(0)
							elif cur == 'C':
								data['Embarked'].append(1)
							else:
								data['Embarked'].append(2)

	return data


def one_hot_encoding(data: dict, feature: str):
	"""
	:param data: dict[str, list], key is the column name, value is its data
	:param feature: str, the column name of interest
	:return data: dict[str, list], remove the feature column and add its one-hot encoding features
	"""
	# Extract the feature column and identify unique values
	data_set = data.pop(feature)
	unique_vals = set(data_set)

	if feature == 'Pclass':
		for val in unique_vals:
			data[f"{feature}_{val - 1}"] = [0] * len(data_set)
		for i, value in enumerate(data_set):
			data[f"{feature}_{value - 1}"][i] = 1

	else:
		for val in unique_vals:
			data[f"{feature}_{val}"] = [0] * len(data_set)
		for i, val in enumerate(data_set):
			data[f"{feature}_{val}"][i] = 1

	return data


def normalize(data: dict):
	"""
	:param data: dict[str, list], key is the column name, value is its data
	:return data: dict[str, list], key is the column name, value is its normalized data
	"""
	for feature in data:
		if feature != 'Sex':
			# Check the range between min and max in features.
			minimum, maximum = min(data[feature]), max(data[feature])
			denominator = maximum - minimum

			data[feature] = [(value - minimum) / denominator for value in data[feature]]

	return data


def learnPredictor(inputs: dict, labels: list, degree: int, num_epochs: int, alpha: float):
	"""
	:param inputs: dict[str, list], key is the column name, value is its data
	:param labels: list[int], indicating the true label for each data
	:param degree: int, degree of polynomial features
	:param num_epochs: int, the number of epochs for training
	:param alpha: float, known as step size or learning rate
	:return weights: dict[str, float], feature name and its weight
	"""

	# Step 1 : Initialize weights
	weights = defaultdict(int)
	keys = list(inputs.keys())
	key_combinations = []

	# Sigmoid function
	def sigmoid(k):
		return 1 / (1 + math.exp(-k))

	# Precompute key combinations for polynomial degree 2
	if degree == 2:
		key_combinations = [(keys[i], keys[j]) for i in range(len(keys)) for j in range(i, len(keys))]

	# Step 2 : Start training
	for epoch in range(num_epochs):
		for index in range(len(labels)):

			# Step 3 : Feature Extract
			phi_x = {}

			for key in keys:
				phi_x[key] = inputs[key][index]

			if degree == 2:
				for m, n in key_combinations:
					phi_x[m + n] = phi_x[m] * inputs[n][index]

			# Step 4 : Update weights
			increment(weights, -alpha * (sigmoid(dotProduct(weights, phi_x)) - labels[index]), phi_x)

	return weights

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import plot_tree


class ModelAnalyzer(object):

	def __init__(self, **kwargs):

		self.data, self.target = self.generate_data(**kwargs)

		self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
			self.data, self.target,
			test_size = self.test_size,
			random_state = self.random_state
		)

	def generate_data(self, data_size, feature_size, n_redundant, n_classes, test_size, random_state):

		self.data_size = data_size
		self.feature_size = feature_size
		self.test_size = test_size
		self.random_state = random_state
		self.n_redundant = n_redundant
		self.n_classes = n_classes

		n_informative = min(10, feature_size - 1)

		return make_classification(
			n_samples = self.data_size,
			n_informative = n_informative,
			n_redundant = self.n_redundant,
			n_features = self.feature_size,
			n_classes = self.n_classes,
			random_state = self.random_state
		)


	def train_knn_model(self, n_neighbors=3):
		self.knn_model = KNeighborsClassifier(n_neighbors=n_neighbors)
		self.knn_model.fit(self.X_train, self.y_train)

	def evaluate_knn_model(self):
		y_pred = self.knn_model.predict(self.X_test)
		accuracy = accuracy_score(self.y_test, y_pred)
		print(f"Точность модели: {accuracy}")

		sns.scatterplot(x = self.X_test[:, 0], y = self.X_test[:, 1], hue = y_pred)
		plt.show()

	def train_decision_tree_model(self):
		self.dt_model = DecisionTreeClassifier(max_depth=None)
		self.dt_model.fit(self.X_train, self.y_train)

	def evaluate_decision_tree_model(self):
		y_pred_dt = self.dt_model.predict(self.X_test)
		accuracy_dt = accuracy_score(self.y_test, y_pred_dt)
		print(f"Точность дерева решений: {accuracy_dt}")

		train_accuracy = accuracy_score(self.y_train, self.dt_model.predict(self.X_train))
		test_accuracy = accuracy_score(self.y_test, y_pred_dt)

		print(f"Точность на обучающей выборке: {train_accuracy}")
		print(f"Точность на тестовой выборке: {test_accuracy}")

		plt.figure(figsize=(18, 12))
		plot_tree(self.dt_model, filled=True, fontsize = 6, max_depth=4)
		plt.title('Визуализация дерева решений')
		plt.show()

	def tune_decision_tree_depth(self, depth_values):
		for depth in depth_values:
			dt_model = DecisionTreeClassifier(max_depth=depth)
			dt_model.fit(self.X_train, self.y_train)
			y_pred_depth = dt_model.predict(self.X_test)
			accuracy_depth = accuracy_score(self.y_test, y_pred_depth)
			print(f"Точность для максимальной глубины={depth}: {accuracy_depth}")

	def save_model(self, model, filename):
		joblib.dump(model, filename)

	def load_model(self, filename):
		loaded_model = joblib.load(filename)
		return loaded_model

	def make_predictions(self, model, X_data, y_true=None):
		predictions = model.predict(X_data)
		accuracy = accuracy_score(y_true, predictions) if y_true is not None else None

		if accuracy is not None:
			print(f"Точность предсказаний для новых данных: {accuracy}")

		return predictions, accuracy

if __name__ == '__main__':
	# Генерация и анализ выборки из 10,000 элементов
	model_analyzer = ModelAnalyzer(data_size=10000, feature_size=20, n_redundant=0, n_classes=2, test_size=0.2, random_state=42)
	model_analyzer.train_knn_model(n_neighbors=3)
	model_analyzer.evaluate_knn_model()

	# Построение дерева решений
	model_analyzer.train_decision_tree_model()

	# Анализ модели
	model_analyzer.evaluate_decision_tree_model()

	# Подбор оптимального max_depth
	model_analyzer.tune_decision_tree_depth(depth_values = [1, 3, 5, 7, 10])

	# Сохранение модели
	model_analyzer.save_model(model_analyzer.dt_model, "decision_tree_model.model")

	# Загрузка модели
	loaded_dt_model = model_analyzer.load_model("decision_tree_model.model")

	# Применение загруженной модели для предсказаний на новых данных
	new_data, new_labels = model_analyzer.generate_data(data_size=10000, feature_size=20, n_redundant=0, n_classes=2, test_size=0.2, random_state=42)
	new_predictions, new_accuracy = model_analyzer.make_predictions(loaded_dt_model, new_data, y_true = new_labels)
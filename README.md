# Model Analyzer

## Использование

1. **Генерация данных и обучение KNN модели**

   ```python
   model_analyzer = ModelAnalyzer(data_size=10000, feature_size=2, test_size=0.2, random_state=42)
   model_analyzer.train_knn_model(n_neighbors=3)
   model_analyzer.evaluate_knn_model()
Построение и анализ модели дерева решений

model_analyzer.train_decision_tree_model()
model_analyzer.evaluate_decision_tree_model()
Подбор оптимальной глубины для дерева решений

model_analyzer.tune_decision_tree_depth(depth_values=[3, 5, 7, 10])
Сохранение и загрузка модели дерева решений

model_analyzer.save_model(model_analyzer.dt_model, "decision_tree_model.model")
loaded_dt_model = model_analyzer.load_model("decision_tree_model.model")
Применение загруженной модели для предсказаний на новых данных

new_data = np.random.rand(1000, 2)
new_labels = np.random.randint(2, size=1000)
new_predictions, new_accuracy = model_analyzer.make_predictions(loaded_dt_model, new_data, y_true=new_labels)

Зависимости
NumPy
Pandas
Matplotlib
Seaborn
Scikit-learn
Joblib
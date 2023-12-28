# DataInsight

## Использование

1. **Генерация данных и обучение KNN модели**

```python
model_analyzer = ModelAnalyzer(data_size=10000, feature_size=2, test_size=0.2, random_state=42)
model_analyzer.train_knn_model(n_neighbors=3)
model_analyzer.evaluate_knn_model()
```
   
2. **Построение и анализ модели дерева решений**
```python
model_analyzer.train_decision_tree_model()
model_analyzer.evaluate_decision_tree_model()
```

3. **Подбор оптимальной глубины для дерева решений**
```python
model_analyzer.tune_decision_tree_depth(depth_values=[3, 5, 7, 10])
```

4. **Сохранение и загрузка модели дерева решений**
```python
model_analyzer.save_model(model_analyzer.dt_model, "decision_tree_model.model")
loaded_dt_model = model_analyzer.load_model("decision_tree_model.model")
```

5. **Применение загруженной модели для предсказаний на новых данных**
```python
new_data = np.random.rand(1000, 2)
new_labels = np.random.randint(2, size=1000)
new_predictions, new_accuracy = model_analyzer.make_predictions(loaded_dt_model, new_data, y_true=new_labels)
```
## Зависимости
```Text
contourpy==1.1.1
cycler==0.12.1
fonttools==4.47.0
importlib-resources==6.1.1
joblib==1.3.2
kiwisolver==1.4.5
matplotlib==3.7.4
numpy==1.24.4
packaging==23.2
pandas==2.0.3
Pillow==10.1.0
pyparsing==3.1.1
python-dateutil==2.8.2
pytz==2023.3.post1
scikit-learn==1.3.2
scipy==1.10.1
seaborn==0.13.0
six==1.16.0
threadpoolctl==3.2.0
tzdata==2023.3
zipp==3.17.0
```



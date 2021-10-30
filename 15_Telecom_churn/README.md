## О проекте

Оператор связи хочет научиться прогнозировать отток клиентов. Если выяснится, что пользователь планирует уйти, ему будут предложены промокоды и специальные условия. Команда оператора собрала персональные данные о некоторых клиентах, информацию об их тарифах и договорах.

Задачи:
- построить модель оттока, с максимальным значением AUC-ROC;
- проверить AUC-ROC на тестовой выборке;
- в качестве дополнительной метрики максимизировать accuracy.

## План проекта

1. Загрузка и анализ общей информации.
2. Преобразование данных.
3. Создание общего датасета.
4. Исследовстельский анализ данных.
5. Обучение моделей (LogisticRegression, DecisionTreeClassifier, RandomForestClassifier, LightGBM).
6. Тестирование лучшей модели.
7. Общий вывод.

## Используемые инструменты и библиотеки

Проект выполнен в [Jupiter Notebook](https://jupyter.org/install.html).

***Формат файла проекта***:

.ipynb

***Библиотеки и методы***:

Для корректного выполнения и анализа файла проекта необходимо импортировать соответствующие библиотеки и методы.

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import warnings

warnings.filterwarnings('ignore')

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
```

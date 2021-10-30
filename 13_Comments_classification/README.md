## О проекте

Пользователи интернет-магазина могут редактировать и дополнять описания товаров, вернее предлагают свои правки и комментируют изменения других.

Необходим инструмент, который будет искать токсичные комментарии и отправлять их на модерацию.

Задача - обучить модель классифицировать комментарии на позитивные и негативные. В наличии - набор данных с разметкой о токсичности правок.

Метрика качества *F1* должна быть не меньше 0.75.


## План проекта

1. Загрузка и изучение общей информации о данных.
2. Анализ данных.
3. Обучение моделей. Подбор гиперпараметров.
4. Проверка на тестовой выборке.
5. Обучение с помощью *BERT*.
6. Общий вывод.

## Используемые инструменты и библиотеки

Проект выполнен в [Jupiter Notebook](https://jupyter.org/install.html).

***Формат файла проекта***:

.ipynb

***Библиотеки и методы***:

Для корректного выполнения и анализа файла проекта необходимо импортировать соответствующие библиотеки и методы.

```python
import numpy as np
import pandas as pd
import torch
import transformers
import nltk
import re
import warnings

from tqdm import notebook
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords as nltk_stopwords
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from catboost import Pool, CatBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import MinMaxScaler
warnings.filterwarnings('ignore')
```

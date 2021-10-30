## О проекте
Необходимо защитить данные клиентов страховой компании.

Задача:
  - разработать метод преобразования данных, чтобы по ним было сложно восстановить персональную информацию.
  - обосновать корректность работы метода.
  - защитить данные, чтобы при преобразовании качество моделей машинного обучения не ухудшилось.

Подбирать наилучшую модель не требуется.

## План проекта

1. Загрузка и изучение общей информации о данных.
2. Подготовка данных.
3. Ответ на вопрос:
 - Признаки умножают на обратимую матрицу. Изменится ли качество линейной регрессии?
    - a. Изменится. Приведсти примеры матриц.
    - b. Не изменится. Указать, как связаны параметры линейной регрессии в исходной задаче и в преобразованной.
5. Анализ недоступных признаков.
6. Предобработка данных.
7. Анализ данных
  - изменение концентрация металлов (Au, Ag, Pb) на различных этапах очистки.
  - сравнение распределения размеров гранул сырья на обучающей и тестовой выборках.
  - исследование суммарной концентрации всех веществ на разных стадиях.
8. Построение модели.
  - функция для вычисления итоговой sMAPE.
  - обучение разных моделей и оценка их качества кросс-валидацией.
  - выбор лучшей модели и её проверка на тестовой выборке.
9. Общий вывод.

## Используемые инструменты и библиотеки

Проект выполнен в [Jupiter Notebook](https://jupyter.org/install.html).

***Формат файла проекта***:

.ipynb

***Библиотеки и методы***:

Для корректного выполнения и анализа файла проекта необходимо импортировать соответствующие библиотеки и методы.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statistics
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from matplotlib.pyplot import show
from scipy.stats import norm
from scipy import stats
```

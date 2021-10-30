## О проекте
Задача - создать прототип модели машинного обучения для компании, разрабатывающей решения для эффективной работы промышленных предприятий.

Модель должна предсказать коэффициент восстановления золота из золотосодержащей руды. В наличии данные с параметрами добычи и очистки.

Модель позволит оптимизировать производство, чтобы не запускать предприятие с убыточными характеристиками.

## План проекта

1. Загрузка и изучение общей информации о данных.
2. Подготовка данных.
3. Вычисление эффективности обогащения. Рассчет *MAE*.
4. Анализ недоступных признаков.
5. Предобработка данных.
6. Анализ данных
7. Расчет рисков и прибыли.
8. Общий вывод.

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

## О проекте
Компания-агрегатор такси собрала исторические данные о заказах такси в аэропортах.

Задача - спрогнозировать (построить модель) количество заказов такси на следующий час, чтобы привлекать больше водителей в период пиковой нагрузки.

Значение метрики RMSE на тестовой выборке должно быть не больше 48.


## План проекта

1. Загрузка и изучение общей информации о данных.
2. Ресемплирование
3. Анализ данных.
4. Обучение моделей. Подбор гиперпараметров.
5. Проверка на тестовой выборке.
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
import matplotlib.pyplot as plt
import seaborn as sns
import statistics
import warnings
warnings.filterwarnings('ignore')

from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.model_selection import train_test_split
```

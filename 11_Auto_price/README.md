## О проекте
Сервис по продаже автомобилей с пробегом разрабатывает приложение для привлечения новых клиентов. В нём можно быстро узнать рыночную стоимость своего автомобиля.

В наличии исторические данные: технические характеристики, комплектации и цены автомобилей.

Задача - построить модель для определения стоимости авто.

Заказчику важны:
- качество предсказания;
- скорость предсказания;
- время обучения.

*Примечание*: для оценки качества моделей использовать метрику *RMSE*.

## План проекта

1. Загрузка и изучение общей информации о данных.
2. Подготовка данных.
3. Обучение разных моделей. Подбор гиперпараметров.
4. Анализ скорости и качества моделей.
5. Общий вывод.

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
import warnings
warnings.filterwarnings('ignore')

from scipy import stats as st
from matplotlib.pyplot import show
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression
```

## О проекте

Из банка стали уходить клиенты. Банковские маркетологи посчитали: сохранять текущих клиентов дешевле, чем привлекать новых.

Задача - спрогнозировать, уйдёт клиент из банка в ближайшее время или нет. В наличии исторические данные о поведении клиентов и расторжении договоров с банком.

Необходимо построить модель с предельно большим значением F1-меры (не менее 0.59). Проверить F1-меру на тестовой выборке.
Дополнительно измерить AUC-ROC, сравнивая её значение с F1-мерой.


## План проекта

1. Загрузка и изучение общей информации о данных.
2. Подготовка данных.
3. Исследование баланса классов. Обучение модели, без учёта дисбаланса.
5. Улучшение качества модели, учитывая дисбаланс классов.
6. Обучение разных модели и определение лучшей.
7. Тестирование.
8. Общий вывод.

## Используемые инструменты и библиотеки

Проект выполнен в [Jupiter Notebook](https://jupyter.org/install.html).

***Формат файла проекта***:

.ipynb

***Библиотеки и методы***:

Для корректного выполнения и анализа файла проекта необходимо импортировать соответствующие библиотеки и методы.

```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')
```

## О проекте

Супермаркет внедряет систему компьютерного зрения для обработки фотографий покупателей. Это поможет определять возраст клиентов, чтобы:
- aнализировать покупки и предлагать товары, которые могут заинтересовать покупателей этой возрастной группы;
- контролировать кассиров при продаже алкоголя.

Задача - построить модель, которая по фотографии определит приблизительный возраст человека.

В наличии - набор фотографий людей с указанием возраста.

Метрикой качества будет *MAE*.

## План проекта

1. Исследовательский анализ набора фото.
2. Подготовка данных.
3. Обучение нейронной сети.
4. Рассчет качества.
5. Общий вывод.

## Используемые инструменты и библиотеки

Проект выполнен в [Jupiter Notebook](https://jupyter.org/install.html).

***Формат файла проекта***:

.ipynb

***Библиотеки и методы***:

Для корректного выполнения и анализа файла проекта необходимо импортировать соответствующие библиотеки и методы.

```python
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
```

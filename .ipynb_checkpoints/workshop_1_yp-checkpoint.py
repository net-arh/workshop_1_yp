#!/usr/bin/env python
# coding: utf-8

# # Анализ стартапов

# ## Описание проекта
# 
# Главная задача - разработать модель для предсказания успешности стартапа (закроется или нет).
# 
# Работа будет проводится с псевдо-реальными (реальные данные в которые добавлена синтетическая составляющая) данными о стартапах, функционировавших в период с 1970 по 2018 годы.
# 
# Выделены следующие этапы работы:
# 
# - загрузка и ознакомление с данными,
# - предварительная обработка,
# - полноценный разведочный анализ,
# - разработка новых синтетических признаков,
# - проверка на мультиколлинеарность,
# - отбор финального набора обучающих признаков,
# - выбор и обучение моделей,
# - итоговая оценка качества предсказания лучшей модели,
# - анализ важности ее признаков,
# - подготовка отчета по исследованию.

# ## Первоначальная настройка

# ### Автоматическая перезагрузка всех модулей

# In[1]:


# загрузка расширения autoreload
get_ipython().run_line_magic('load_ext', 'autoreload')

# автоматическая перезагрузка всех модулей
get_ipython().run_line_magic('autoreload', '2')


# ### Установка всех необходимых пакетов

# In[2]:


# для определения правильности написания стран
get_ipython().system('pip install pycountry -q')

# для определения правильности написания штатов
get_ipython().system('pip install us -q')

# для определения широты и долготы городов
get_ipython().system('pip install geopy')


# ### Подключение всех необходимых библиотек

# In[3]:


# для взаимодействия с операционной системой
import os

# для работы с регулярными выражениями
import re

# для работы со временем
import time

# для управления предупреждениями
import warnings

# для подсчета частоты категорий
from collections import Counter

# для работы с векторами, матрицами и многомерными массивами
import numpy as np

# для работы с данными
import pandas as pd

# для статистических вычислений
import scipy.stats as pt

# для визуализации данных
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import plotly.express as px

# для разделения данных и поиска гиперпараметров
from sklearn.model_selection import (train_test_split,
                                     RandomizedSearchCV,
                                     GridSearchCV)

# для построения пайплайнов и трансформеров
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# для кодирования категориальных данных и нормализации числовых данных
from sklearn.preprocessing import (OneHotEncoder, OrdinalEncoder,
                                   LabelEncoder, StandardScaler,
                                   MinMaxScaler, RobustScaler)

# для заполнения пропущенных значений
from sklearn.impute import SimpleImputer

# для векторизации текста
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer

# для устранение дисбаланса классов
from imblearn.over_sampling import SMOTE

# модели машинного обучения
from sklearn.linear_model import (LogisticRegression, LinearRegression,
                                  Ridge, Lasso, RidgeClassifier)
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

# метрики оценки качества моделей
from sklearn.metrics import (confusion_matrix, accuracy_score,
                             precision_score, recall_score,
                             precision_recall_curve, roc_auc_score,
                             make_scorer, r2_score, mean_squared_error,
                             mean_absolute_error)

# dummy модели
from sklearn.dummy import DummyRegressor

# для вычисления коэффициента VIF
from statsmodels.stats.outliers_influence import variance_inflation_factor

# для визуализации пропущенных данных
import missingno as msno

# для автоматического подбора гиперпараметров
import optuna
from optuna.integration import OptunaSearchCV
from optuna.distributions import (CategoricalDistribution, FloatDistribution,
                                  IntDistribution)

# для объяснения предсказаний
import shap

# для отображения контента в jupyter notebook
from IPython.display import Markdown

# для работы с кодами стран и их регионов
import pycountry

# для работы с кодами штатов
import us

# для получение широты и долготы городов
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut


# ### Задание всех необходимых опций

# In[4]:


# настройка pandas для отображения всех столбцов и строк
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# установка палитры
sns.set_palette('muted')

# создание констант для
# воспроизводимости результатов
RANDOM_STATE = 42

# объект геокодера
geolocator = Nominatim(user_agent='geoapi', timeout=10)

# кэш для хранения координат
city_cache = {}

# подавление предупреждений об экспериментальных функциях
warnings.filterwarnings('ignore', category=optuna.exceptions.ExperimentalWarning)

# установить уровень логирования на WARNING,
# чтобы уменьшить количество выводимой информации
optuna.logging.set_verbosity(optuna.logging.WARNING)

# подавление всех предупреждений
# warnings.filterwarnings('ignore')


# ### Функции

# #### `create_dataframe()`

# In[5]:


def create_dataframe(pth1, pth2='', sep=',', decimal='.', encoding='utf-8'):
    '''
    Функция возвращает датафрейм по заданным путям pth1
    и pth2 к датасету.
    
    Аргументы функции: 
    - pth1: путь к датасету (строка),
    - pth2: альтернативный путь
      к датасету (строка),
    - sep: разделитель столбцов (строка),
    - decimal: десятичный разделитель (строка).
    '''
    if os.path.exists(pth1):      
        return pd.read_csv(pth1, sep=sep, decimal=decimal, encoding=encoding)
    elif os.path.exists(pth2):
        return pd.read_csv(pth2, sep=sep, decimal=decimal, encoding=encoding)
    else:
        print('Ошибка чтения')


# #### `data_review()`

# In[6]:


def data_review(df):
    '''
    Функция выводит информацию
    о заданном датафрейме df.
    
    Аргументы функции:
    - df: заданный датафрейм.
    '''
    # вывод первых пяти строк данных
    display(Markdown('##### Первые 5 строк датасета'))
    display(df.head())
    
    # вывод информации о столбцах и типах данных
    display(Markdown('##### Информация о столбцах и типах данных'))
    print(df.info())
    
    # вывод статистического описания
    display(Markdown('##### Статистическое описание'))
    display(df.describe(include='all'))


# #### `change_data_types()`

# In[7]:


def change_data_types(df, column_types):
    '''
    Аргументы функции:
    - df: выбранный датафрейм,
    - column_types: заданный словарь
      (ключ - название столбца,
       значение - нужный тип данных).
                       
    Функция изменяет типы данных столбцов выбранного
    датафрейма df согласно заданному словарю column_types.
    '''
    # изменение типов данных столбцов
    for col, col_type in column_types.items():
        try:
            if col_type == 'datetime':
                df[col] = pd.to_datetime(df[col], format='%Y-%m-%d')
            else:
                df[col] = df[col].astype(col_type)
        except ValueError:
            print(f'Ошибка преобразования столбца "{col}" в тип {col_type}')
        except KeyError:
            print(f'Столбец "{col}" отсутствует в DataFrame')
    
    # проверка
    print(df.dtypes)
    
    # возвращение заданного датафрейма
    return df


# #### `fill_nan()`

# In[8]:


def fill_nan(df, fillna_pipeline=None, num_cols=[], cat_cols=[], obj_cols=[]):
    '''
    Функция заполняет пропущенные значения в заданном
    датафрейме, возвращая его измененную копию и пайплайн.
    
    Аргументы функции:
    - df: заданный датафрейм,
    - fillna_pipeline: пайплайн по заполнению
    пропущенных значений,
    - num_cols: числовые столбцы (список),
    - cat_cols: категориальные столбцы (список).
    '''
    # проверка на наличие всех необходимых столбцов в датафрейме
    missing_num_cols = [col for col in num_cols if col not in df.columns]
    missing_cat_cols = [col for col in cat_cols if col not in df.columns]
    missing_obj_cols = [col for col in obj_cols if col not in df.columns]
    
    # добавление отсутствующих столбцов
    for col in missing_num_cols:
        df[col] = np.nan
    for col in missing_cat_cols:
        df[col] = np.nan
    for col in missing_obj_cols:
        df[col] = np.nan
    
    if fillna_pipeline is None:
        
        # создание пайплайна по заполнению пропущенных значений
        fillna_pipeline = ColumnTransformer(
            [
                (
                    'num',
                    SimpleImputer(missing_values=np.nan, strategy='median'),
                    num_cols
                ),
                (
                    'cat',
                    SimpleImputer(missing_values=np.nan, strategy='constant', fill_value='unknown'),
                    cat_cols
                ),
                (
                    'obj',
                    SimpleImputer(missing_values=np.nan, strategy='constant', fill_value='unknown'),
                    obj_cols
                )
            ],
            remainder='passthrough'
        )
        
        # заполнение пропущенных значений
        df_np = fillna_pipeline.fit_transform(df)
    
    else:
        
        # заполнение пропущенных значений
        df_np = fillna_pipeline.transform(df)
        
    # получение имен столбцов, которые прошли через remainder='passthrough'
    passthrough_cols = [col for col in df.columns if col not in num_cols + cat_cols + obj_cols]
    
    # объединение всех имен столбцов
    all_cols = num_cols + cat_cols + obj_cols + passthrough_cols
    
    # преобразование обратно в датафрейм
    df_filled = pd.DataFrame(df_np, columns=all_cols)
    
    # восстановление типов данных
    for col in num_cols:
        df_filled[col] = df_filled[col].astype(df[col].dtype)
    for col in cat_cols:
        df_filled[col] = df_filled[col].astype('category')
        if 'unknown' not in df_filled[col].cat.categories:
            df_filled[col] = df_filled[col].cat.add_categories(['unknown'])
    for col in obj_cols:
        df_filled[col] = df_filled[col].astype(df[col].dtype)
    
    # восстановление исходного порядка столбцов
    df_filled = df_filled[df.columns]
    
    # удаление временно добавленных столбцов
    df_filled.drop(
        columns=missing_num_cols + missing_cat_cols + missing_obj_cols,
        inplace=True
    )
    
    # возвращение датафрейма и пайплайна
    return df_filled, fillna_pipeline


# #### `cat_unique()`

# In[9]:


def cat_unique(df):
    '''
    Функция выводит все уникальные значения по столбцам
    типа category заданного датафрейма df.
    
    Аргументы функции:
    - df: заданный датафрейм.
    '''
    for column in df.select_dtypes(include=['category']).columns:
        print(f'\nТоп-50 значений для столбца: {column}\n{"-"*50}')
        
        try:
            # подсчет количества вхождений уникальных значений
            category_counts = df[column].value_counts().head(30)

            # преобразование в DataFrame для удобства
            category_counts_df = category_counts.reset_index()
            category_counts_df.columns = ['category', 'frequency']

            # вывод результата
            display(category_counts_df)

        except Exception as e:
            print(f"Ошибка при обработке столбца {column}: {e}")


# #### `categorize()`

# In[10]:


def categorize(row):
    '''
    
    '''
    # словарь для выделения более крупных категорий
    category_mapping = {
        # IT
        'Software': 'IT',
        'Mobile': 'IT',
        'Enterprise Software': 'IT',
        'Analytics': 'IT',
        'Hardware': 'IT',
        'Hardware + Software': 'IT',
        'Cloud-Based Music': 'IT',
        'Cloud Computing': 'IT',
        'Communications Hardware': 'IT',
        'SaaS': 'IT',
        'Apps': 'IT',
        'Facebook Applications': 'IT',
        'Big Data': 'IT',
        'Big Data Analytics': 'IT',
        'Web Hosting': 'IT',
        'Web Development': 'IT',
        'Android': 'IT',
        'iOS': 'IT',
        'CRM': 'IT',
        'Information Technology': 'IT',
        'Health Care Information Technology': 'IT',
        'IT': 'IT',
        # медицина
        'Biotechnology': 'Medicine',
        'Health Care': 'Medicine',
        'Health Diagnostics': 'Medicine',
        'Medical': 'Medicine',
        'Medical Devices': 'Medicine',
        'Quantified Self': 'Medicine',
        'Health and Wellness': 'Medicine',
        'Pharmaceuticals': 'Medicine',
        'Medicine': 'Medicine',
        # медиа
        'Curated Web': 'Media',
        'Social Media': 'Media',
        'Advertising': 'Media',
        'Video': 'Media',
        'Video Streaming': 'Media',
        'Music': 'Media',
        'Photography': 'Media',
        'News': 'Media',
        'Social Network Media': 'Media',
        'Search': 'Media',
        'Networking': 'Media',
        'Messaging': 'Media',
        'Digital Media': 'Media',
        'Content': 'Media',
        'Publishing': 'Media',
        'Reviews and Recommendations': 'Media',
        'Public Relations': 'Media',
        'Media': 'Media',
        # продажи
        'E-Commerce': 'Sales',
        'Mobile Commerce': 'Sales',
        'Social Commerce': 'Sales',
        'Retail': 'Sales',
        'Marketplaces': 'Sales',
        'Online Shopping': 'Sales',
        'Sales and Marketing': 'Sales',
        'Brand Marketing': 'Sales',
        'Internet Marketing': 'Sales',
        'Sales': 'Sales',
        # развлечения
        'Games': 'Entertainment',
        'Sports': 'Entertainment',
        'Events': 'Entertainment',
        'Entertainment': 'Entertainment',
        # энергетика
        'Clean Technology': 'Energy',
        'Electric Vehicles': 'Energy',
        'Energy': 'Energy',
        # финансы
        'Financial Services': 'Finance',
        'FinTech': 'Finance',
        'Payments': 'Finance',
        'Finance': 'Finance',
        # путешествия
        'Transportation': 'Travel',
        'Travel': 'Travel',
        # услуги
        'Consulting': 'Services',
        'Location Based Services': 'Services',
        'Service Providers': 'Services',
        'Customer Services': 'Services',
        'Services': 'Services',
        # недвижимость
        'Hospitality': 'Real Estate',
        'Real Estate': 'Real Estate',
        # мода
        'Fashion': 'Fashion',
        # фитнес
        'Fitness': 'Fitness',
        # образование
        'Education': 'Education',
        # дизайн
        'Design': 'Design',
        # промышленность
        'Manufacturing': 'Industry',
        'Semiconductors': 'Industry',
        'Automotive': 'Industry',
        'Industry': 'Industry',
        # еда
        'Restaurants': 'Food',
        'Food': 'Food',
        # безопасность
        'Security': 'Security',
        # коллаборация
        'Collaboration': 'Collaboration',
        # HR
        'Recruiting': 'HR',
        'Human Resources': 'HR',
        'HR': 'HR',
        # технологии
        'Internet': 'Technology',
        'Internet of Things': 'Technology',
        'Telecommunications': 'Technology',
        'EdTech': 'Technology',
        'Wireless': 'Technology',
        'iPhone': 'Technology',
        'Technology': 'Technology',
        # потребительское
        'Consumer Goods': 'Consumer',
        'Consumer Electronics': 'Consumer',
        'Consumer': 'Consumer',
        # бизнес
        'Startups': 'Business',
        'Business Services': 'Business',
        'Business Intelligence': 'Business',
        'Enterprises': 'Business',
        'B2B': 'Business',
        'Crowdsourcing': 'Business',
        'Small and Medium Businesses': 'Business',
        'Business': 'Business',
        # настоящее время
        'Real Time': 'Real Time',
        # образ жизни
        'Lifestyle': 'Lifestyle',
        # неизвестное
        'unknown': 'unknown',
        # разное
        'Other': 'Other'
    }

    # укрупнение категорий
    categories = row.split('|') if pd.notna(row) else []
    main_groups = set()
    for cat in categories:
        if cat in category_mapping:
            main_groups.add(category_mapping[cat])
    
    # возвращение категории
    return '|'.join(sorted(main_groups)) if main_groups else 'Other'


# #### `check_abbr()`

# In[11]:


def check_abbr(df, col, valid):
    '''
    Функция выводит кол-во некорректных аббревиатур
    в выбранном столбце заданного датафрейма, при этом
    заменяя некорректные значения на 'unknown'. Также
    выводятся все уникальные значения.
    
    Аргументы функции:
    - df: заданный датафрейм,
    - col: выбранный столбец,
    - valid: список корректных значений.
    '''
    # определение некорректных значений
    invalid_mask = ~df[col].isin(valid)
    invalid_count = invalid_mask.sum()
    
    # замена некорректных значений на 'unknown'
    df[col] = df[col].apply(
        lambda x: x if x in valid else 'unknown'
    )
    
    # проверка
    print(f'Встречено {invalid_count} некорректных аббревиатур\n')
    print('Уникальные значения:')
    print(df[col].unique())


# #### `obvious_duplicates()`

# In[12]:


def obvious_duplicates(df, comb_cols=[]):
    '''
    Функция выводит количество явных дубликатов.
    
    Аргументы функции:
    - df: заданный датафрейм,
    - comb_cols: комбинации столбцов,
      по которым нужно проверить наличие
      явных дубликатов (двумерный список).
    '''
    # вывод столбцов и кол-ва дубликатов,
    # находящихся в этих столбцах
    for column in df.columns:
        print(f'{column}: {df.duplicated(subset=column).sum()}')
    
    # отступ
    if comb_cols != []:
        print()
    
    # кол-во дубликатов по комбинациям столбцов
    for column in comb_cols:
        print(f'{column}: {df.duplicated(subset=column).sum()}')
    
    # вывод кол-ва явных дубликатов всего датафрейма
    print(f'\nкол-во абсолютных явных дубликатов: {df.duplicated().sum()}')


# #### `more_describe()`

# In[13]:


def more_describe(df, col=None, hue=None):
    '''
    Функция выводит расширенное статистическое описание
    заданного числового столбца выбранного датафрейма.
    
    Аргументы функции:
    - df: выбранный датафрейм,
    - column: заданный числовой столбец (строка),
    - hue: дополнительная разбивка (строка).
    '''
    if hue is not None:
        
        description = pd.DataFrame(columns=df[hue].unique())
        
        # перебор по категориям
        for category in df[hue].unique():
            
            # фильтрация данных по категории
            category_data = df[df[hue] == category]
            
            # получение стандартного описания
            description[category] = category_data[col].describe()
            
            # вычисление квартилей и межквартильного размаха
            # для каждой категории
            q1 = category_data[col].quantile(q=0.25)
            q3 = category_data[col].quantile(q=0.75)
            iqr = q3 - q1
            
            # определение выбросов
            ems = ((category_data[col] < (q1 - 1.5 * iqr)) | 
                   (category_data[col] > (q3 + 1.5 * iqr)))
            
            # подсчет выбросов и их характеристик
            count = category_data[ems].shape[0]
            share = count / category_data[col].count() * 100
            min_em = category_data[ems][col].min()
            max_em = category_data[ems][col].max()
            
            # подготовка результатов
            description.loc['iqr', category] = iqr
            description.loc['lower_bound', category] = q1
            description.loc['upper_bound', category] = q3
            description.loc['outliers_count', category] = count
            description.loc['outliers_pct', category] = share
            description.loc['outlier_min', category] = min_em
            description.loc['outlier_max', category] = max_em
    
    else:
        
        if col is not None:
            
            # получение стандартного описания
            description = df[col].describe()
            
            # вычисление квартилей и межквартильного размаха
            q1 = df[col].quantile(q=0.25)
            q3 = df[col].quantile(q=0.75)
            iqr = q3 - q1
            
            # определение выбросов
            ems = ((df[col] < (q1-1.5*iqr)) |\
                   (df[col] > (q3+1.5*iqr)))
            
            # подсчет выбросов и их характеристик
            count = df[ems][col].count()
            share = count / df[col].count() * 100
            min_em = df[ems][col].min()
            max_em = df[ems][col].max()
            
        else:
            
            # получение стандартного описания
            description = df.describe()
            
            # вычисление квартилей и межквартильного размаха
            q1 = df.quantile(q=0.25)
            q3 = df.quantile(q=0.75)
            iqr = q3 - q1
            
            # определение выбросов
            ems = ((df < (q1-1.5*iqr)) |\
                   (df > (q3+1.5*iqr)))
            
            # подсчет выбросов и их характеристик
            count = df[ems].count()
            share = count / df.count() * 100
            min_em = df[ems].min()
            max_em = df[ems].max()
        
        # подготовка результатов
        description['iqr'] = iqr
        description['lower_bound'] = q1
        description['upper_bound'] = q3
        description['outliers_count'] = count
        description['outliers_pct'] = share
        description['outliers_min'] = min_em
        description['outliers_max'] = max_em
    
    return description


# #### `box_plots()`

# In[14]:


def box_plots(df, t=''):
    '''
    Функция составляет график ящиков с усами для всех числовых столбцов
    заданного датафрейма df.
    
    Аргументы функции:
    - df: заданный датафрейм,
    - t: название графика (строка).
    '''
    # выбор числовых столбцов
    numerical_df = df.select_dtypes(include='number')
    
    # преобразование DataFrame в long format
    df_long = numerical_df.melt(var_name='столбец', value_name='значение')
    
    # построение ящиков с усами
    plt.figure(figsize=(15, 10))
    sns.boxplot(data=df_long, x='столбец', y='значение')
    
    # создание подписей над осями графика
    plt.xlabel('столбец', labelpad=15, fontsize=12)
    plt.ylabel('значение', labelpad=15, fontsize=12)
    
    # заголовок
    plt.title(t, pad=10, fontsize=18)
    
    # отображение графика
    plt.show()


# #### `tree_map()`

# In[15]:


def tree_map(df, cat, subcat=None, subsubcat=None, t='', l=None, figsize=(1800, 1000)):
    '''
    Функция составляет treemap по заданному DataFrame.

    Аргументы функции:
    - df: DataFrame,
    - cat: столбец с родительскими категориями,
    - subcat: столбец с дочерними категориями (необязательно),
    - t: название графика,
    - l: словарь подписей осей.
    '''
    # преобразование категориальных столбцов в строки (object), чтобы ускорить groupby
    df_temp = df.copy()
    columns_to_convert = [col for col in [cat, subcat, subsubcat] if col is not None]
    for col in columns_to_convert:
        df_temp[col] = df_temp[col].astype(str)

    # группировка
    if subsubcat is not None and subcat is not None:
        frequency = df_temp.groupby([cat, subcat, subsubcat], observed=False).size().reset_index(name='count')
        path = [cat, subcat, subsubcat]
    elif subcat is not None:
        frequency = df_temp.groupby([cat, subcat], observed=False).size().reset_index(name='count')
        path = [cat, subcat]
    else:
        frequency = df_temp[cat].value_counts().reset_index()
        frequency.columns = [cat, 'count']
        path = [cat]

    # вычисление процентов
    frequency['percent'] = (frequency['count'] / frequency['count'].sum()) * 100

    # построение treemap
    fig = px.treemap(
        frequency, 
        path=path, 
        values='count', 
        title=t,
        labels=l,
        hover_data={'percent': ':.2f'}
    )

    # задание размеров графика
    fig.update_layout(width=figsize[0], height=figsize[1])

    # отображение графика
    fig.show()


# #### `hist_plot()`

# In[16]:


def hist_plot(df, x=None, hue=None, bins=None, xlim=None, xl='', yl='', lt='', t='', figsize=(1800, 800)):
    '''
    Функция строит гистограмму с ящиком с усами (box plot)
    и разбивкой по целевому категориальному признаку.

    Аргументы функции:
    - df: DataFrame с данными,
    - x: столбец с значениями по оси X,
    - hue: дополнительная разбивка,
    - xl: подпись оси X,
    - yl: подпись оси Y,
    - t: заголовок графика.
    '''
    # построение гистограммы
    fig = px.histogram(
        df,
        x=x,
        color=hue,
        nbins=bins,
        range_x=xlim,
        marginal='box',
        opacity=0.5,
        barmode='group',
        title=t
    )

    if hue is not None:
        
        # заголовок легенды
        fig.update_layout(legend_title_text=lt)
        
    # названия осей
    fig.update_layout(
        xaxis_title=xl,
        yaxis_title=yl
    )
    
    # подписи при наведении
    fig.update_traces(
        hovertemplate='Значение: %{x}<br>Частота: %{y}<br>'
    )
    
    # задание размеров графика
    fig.update_layout(width=figsize[0], height=figsize[1])
    
    # отображение графика
    fig.show()


# #### `pie_plot()`

# In[17]:


def pie_plot(df, col, lt='', t='', figsize=(1800, 800)):
    '''
    Функция составляет круговую диаграмму
    по заданному датафрейму.
    
    Аргументы функции:
    - df: заданный датафрейм,
    - col: столбец с категор. значениями (строка),
    - lt: название легенды (строка),
    - t: название графика (строка).
    '''
    # построение круговой диаграммы
    fig = px.pie(
        df, 
        names=col,
        title=t
    )
    
    # заголовок легенды
    fig.update_layout(legend_title_text=lt)
    
    # подписи при наведении
    fig.update_traces(
        textinfo='percent+label',
        hovertemplate='Категория: %{label}<br>Частота: %{value}<br>Процент: %{percent}'
    )
    
    # задание размеров графика
    fig.update_layout(width=figsize[0], height=figsize[1])
    
    # отображение графика
    fig.show()


# #### `vif_heatmap()`

# In[18]:


def vif_heatmap(df, t='', figsize=(1800, 800)):
    '''
    Функция создает тепловую карту на основе коэффициента VIF.

    Аргументы:
    - df: датафрейм с числовыми признаками (без категориальных),
    - t: название графика (строка),
    - figsize: размер графика (по умолчанию (1800, 800)).
    '''
    # вычисление VIF для каждого признака
    vif_data = pd.DataFrame()
    vif_data['feature'] = df.columns
    vif_data['vif'] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]

    # создание DataFrame для тепловой карты
    vif_matrix = pd.DataFrame(np.diag(vif_data['vif']), index=df.columns, columns=df.columns)

    # создание тепловой карты VIF
    fig = px.imshow(
        vif_matrix,
        color_continuous_scale='Reds',
        title=t
    )

    # изменение размеров графика
    fig.update_layout(width=figsize[0], height=figsize[1])

    # отображение графика
    fig.show()


# #### `get_lat_lon()`

# In[19]:


def get_lat_lon(city):
    """Получает широту и долготу города, используя кэш"""
    if pd.isna(city):  # Пропускаем NaN
        return None, None

    if city in city_cache:  # Если город уже есть в кеше, используем его
        return city_cache[city]
    
    try:
        location = geolocator.geocode(city)
        if location:
            city_cache[city] = (location.latitude, location.longitude)
            return location.latitude, location.longitude
        else:
            return None, None
    except GeocoderTimedOut:
        return None, None


# #### `get_coords_city()`

# In[20]:


def get_coords_city(df, col, path_to_save):
    '''
    '''
    # получение уникальных городов
    unique_cities = df[col].dropna().unique()
    
    # запрос координат только для уникальных городов
    city_coords = {city: get_lat_lon(city) for city in unique_cities}
    
    # применение к датафрейму
    df[['latitude', 'longitude']] = df[col].map(city_coords).apply(pd.Series)
    
    # сохранение результатов
    df.to_csv(path_to_save, index=False)
    
    # проверка
    df[[col, 'latitude', 'longitude']].head(10)


# #### `optuna_val()`

# In[21]:


def optuna_val(pipe, param_distributions, scoring,
               X_tr, y_tr, n_trials=50, cv=5, text=''):
    '''
    Функция проводит кросс-валидацию с помощью OptunaSearchCV
    и возвращает его объект.
    
    Аргументы функции:
    - pipe: заданный пайплайн,
    - param_distributions: выбранные гиперпараметры,
    - scoring: метрика для оценки качества моделей,
    - X_tr: входные признаки тренировочной выборки,
    - y_tr: целевой признак тренировочной выборки,
    - n_trials: кол-во итераций посика,
    - cv: кол-во фолдов,
    - text: текст при выводе результатов.
    '''
    # поиск оптимальных гиперпараметров
    optuna_search = OptunaSearchCV(
        estimator=pipe,
        param_distributions=param_distributions,
        scoring=scoring,
        n_trials=n_trials,
        cv=cv,
        random_state=42
    )
    optuna_search.fit(X_tr, y_tr)
    
    # сохранение результатов кросс-валидации
    best_params = optuna_search.best_params_
    metric_val = abs(np.round(optuna_search.best_score_, 2))
    
    # создание и обучение лучшей модели с найденными параметрами
    best_model = clone(pipe).set_params(**best_params)
    start_time_train = time.time()
    best_model.fit(X_tr, y_tr)
    end_time_train = time.time()
    
    # вычисление времени обучения
    time_train = round(end_time_train - start_time_train, 2)
    
    # предсказание на валидационных данных
    start_time_pred = time.time()
    y_pred = best_model.predict(X_tr)
    end_time_pred = time.time()
    
    # вычисление скорости предсказания
    pred_speed = round(len(X_tr) / (end_time_pred - start_time_pred), 2)
    
    # Вывод результатов
    print(text)
    print(f'Метрика на валидационных данных: {metric_val}')
    print(f'Время обучения лучшей модели: {time_train} сек.')
    print(f'Скорость предсказания лучшей модели: {pred_speed} предск./сек.')
    print(f'Гиперпараметры: {best_params}\n')
    
    # возвращение объекта optuna
    return optuna_search


# ## Загрузка данных

# ### Тренировочный датасет

# In[22]:


# создание датафрейма
startups_train = create_dataframe('datasets/kaggle_startups_train_28062024.csv')

# обзор данных
data_review(startups_train)


# **Вывод:**
# 
# Создан тренировочный датафрейм (`startups_train`), содержащий в себе данные о стартапах, которые будут использоваться в качестве обучающих данных. Он имеет 13 столбцов и 52 516 строк. При этом столбцы имеют следующий характер:
# 
# - `name` — название стартапа. Имеет тип `object` и 52 515 уникальных значений;
# - `category_list` — список категорий, к которым относится стартап. Имеет тип `object` и 22 105 уникальных значений. При этом наиболее часто встречающиеся значение `Software` (встречается 3 207 раз);
# - `funding_total_usd` — общая сумма финансирования в USD. Имеет тип `float64`, а значения распределены примерно от 1 до 30 079 500 000 USD. При этом среднее значение составляет примерно 18 247 480 USD, а медиана — 2 000 000 USD;
# - `status` — статус стартапа (закрыт или действующий). Имеет тип `object` и 2 уникальных значения. При этом наиболее часто встречающиеся значение `operating` (встречается 47 599 раз);
# - `country_code` — код страны. Имеет тип `object` и 134 уникальных значений. При этом наиболее часто встречающиеся значение `USA` (встречается 29 702 раза);
# - `state_code` — код штата. Имеет тип `object` и 300 уникальных значений. При этом наиболее часто встречающиеся значение `CA` (встречается 10 219 раз);
# - `region` — регион. Имеет тип `object` и 1 036 уникальных значений. При этом наиболее часто встречающиеся значение `SF Bay Area` (встречается 6 970 раз);
# - `city` — город. Имеет тип `object` и 4 477 уникальных значений. При этом наиболее часто встречающиеся значение `San Francisco` (встречается 2 824 раза);
# - `funding_rounds` — количество раундов финансирования. Имеет тип `int64`, а значения распределены от 1 до 19 раундов. При этом среднее значение составляет примерно 1.74, а медиана — 1;
# - `founded_at` — дата основания. Имеет тип `object` и 5 402 уникальных значения. При этом наиболее часто встречающиеся значение `2012-01-01` (встречается 2 171 раз);
# - `first_funding_at` — дата первого раунда финансирования. Имеет тип `object` и 4 603 уникальных значения. При этом наиболее часто встречающиеся значение `2013-01-01` (встречается 450 раз);
# - `last_funding_at` — дата последнего раунда финансирования. Имеет тип `object` и 4 305 уникальных значений. При этом наиболее часто встречающиеся значение `2014-01-01` (встречается 369 раз);
# - `closed_at` — дата закрытия стартапа (если применимо). Имеет тип `object` и 3 008 уникальных значений. При этом наиболее часто встречающиеся значение `2016-12-02` (встречается 8 раз).

# ### Тестовый датасет

# In[23]:


# создание датафрейма
startups_test = create_dataframe('datasets/kaggle_startups_test_28062024.csv')

# обзор данных
data_review(startups_test)


# **Вывод:**
# 
# Создан тестовый датафрейм (`startups_test`), содержащий в себе данные о стартапах, которые будут использоваться в качестве тестовых данных. Он имеет 11 столбцов и 13 125 строк. При этом столбцы имеют следующий характер:
# 
# - `name` — название стартапа. Имеет тип `object` и 13 125 уникальных значений;
# - `category_list` — список категорий, к которым относится стартап. Имеет тип `object` и 6 206 уникальных значений. При этом наиболее часто встречающиеся значение `Software` (встречается 775 раз);
# - `funding_total_usd` — общая сумма финансирования в USD. Имеет тип `float64`, а значения распределены примерно от 1 до 4 715 000 000 USD. При этом среднее значение составляет примерно 16 549 100 USD, а медиана — 2 000 000 USD;
# - `country_code` — код страны. Имеет тип `object` и 96 уникальных значений. При этом наиболее часто встречающиеся значение `USA` (встречается 7 428 раз);
# - `state_code` — код штата. Имеет тип `object` и 235 уникальных значений. При этом наиболее часто встречающиеся значение `CA` (встречается 2 552 раза);
# - `region` — регион. Имеет тип `object` и 688 уникальных значений. При этом наиболее часто встречающиеся значение `SF Bay Area` (встречается 1 750 раз);
# - `city` — город. Имеет тип `object` и 2 117 уникальных значений. При этом наиболее часто встречающиеся значение `San Francisco` (встречается 656 раз);
# - `funding_rounds` — количество раундов финансирования. Имеет тип `int64`, а значения распределены от 1 до 15 раундов. При этом среднее значение составляет примерно 1.71, а медиана — 1;
# - `first_funding_at` — дата первого раунда финансирования. Имеет тип `object` и 3 299 уникальных значений. При этом наиболее часто встречающиеся значение `2014-01-01` (встречается 126 раз);
# - `last_funding_at` — дата последнего раунда финансирования. Имеет тип `object` и 3 021 уникальных значения. При этом наиболее часто встречающиеся значение `2014-01-01` (встречается 111 раз);
# - `lifetime` — время существования стартапа в днях. Имеет тип `int64`, а значения распределены от 52 до 17 167 дней. При этом среднее значение составляет примерно 3097.61, а медиана — 2 526.

# ### Вывод загрузки данных
# 
# #### Тренировочный датасет
# 
# Создан тренировочный датафрейм (`startups_train`), содержащий в себе данные о стартапах, которые будут использоваться в качестве обучающих данных. Он имеет 13 столбцов и 52 516 строк. При этом столбцы имеют следующий характер:
# 
# - `name` — название стартапа. Имеет тип `object` и 52 515 уникальных значений;
# - `category_list` — список категорий, к которым относится стартап. Имеет тип `object` и 22 105 уникальных значений. При этом наиболее часто встречающиеся значение `Software` (встречается 3 207 раз);
# - `funding_total_usd` — общая сумма финансирования в USD. Имеет тип `float64`, а значения распределены примерно от 1 до 30 079 500 000 USD. При этом среднее значение составляет примерно 18 247 480 USD, а медиана — 2 000 000 USD;
# - `status` — статус стартапа (закрыт или действующий). Имеет тип `object` и 2 уникальных значения. При этом наиболее часто встречающиеся значение `operating` (встречается 47 599 раз);
# - `country_code` — код страны. Имеет тип `object` и 134 уникальных значений. При этом наиболее часто встречающиеся значение `USA` (встречается 29 702 раза);
# - `state_code` — код штата. Имеет тип `object` и 300 уникальных значений. При этом наиболее часто встречающиеся значение `CA` (встречается 10 219 раз);
# - `region` — регион. Имеет тип `object` и 1 036 уникальных значений. При этом наиболее часто встречающиеся значение `SF Bay Area` (встречается 6 970 раз);
# - `city` — город. Имеет тип `object` и 4 477 уникальных значений. При этом наиболее часто встречающиеся значение `San Francisco` (встречается 2 824 раза);
# - `funding_rounds` — количество раундов финансирования. Имеет тип `int64`, а значения распределены от 1 до 19 раундов. При этом среднее значение составляет примерно 1.74, а медиана — 1;
# - `founded_at` — дата основания. Имеет тип `object` и 5 402 уникальных значения. При этом наиболее часто встречающиеся значение `2012-01-01` (встречается 2 171 раз);
# - `first_funding_at` — дата первого раунда финансирования. Имеет тип `object` и 4 603 уникальных значения. При этом наиболее часто встречающиеся значение `2013-01-01` (встречается 450 раз);
# - `last_funding_at` — дата последнего раунда финансирования. Имеет тип `object` и 4 305 уникальных значений. При этом наиболее часто встречающиеся значение `2014-01-01` (встречается 369 раз);
# - `closed_at` — дата закрытия стартапа (если применимо). Имеет тип `object` и 3 008 уникальных значений. При этом наиболее часто встречающиеся значение `2016-12-02` (встречается 8 раз).
# 
# #### Тестовый датасет
# 
# Создан тестовый датафрейм (`startups_test`), содержащий в себе данные о стартапах, которые будут использоваться в качестве тестовых данных. Он имеет 11 столбцов и 13 125 строк. При этом столбцы имеют следующий характер:
# 
# - `name` — название стартапа. Имеет тип `object` и 13 125 уникальных значений;
# - `category_list` — список категорий, к которым относится стартап. Имеет тип `object` и 6 206 уникальных значений. При этом наиболее часто встречающиеся значение `Software` (встречается 775 раз);
# - `funding_total_usd` — общая сумма финансирования в USD. Имеет тип `float64`, а значения распределены примерно от 1 до 4 715 000 000 USD. При этом среднее значение составляет примерно 16 549 100 USD, а медиана — 2 000 000 USD;
# - `country_code` — код страны. Имеет тип `object` и 96 уникальных значений. При этом наиболее часто встречающиеся значение `USA` (встречается 7 428 раз);
# - `state_code` — код штата. Имеет тип `object` и 235 уникальных значений. При этом наиболее часто встречающиеся значение `CA` (встречается 2 552 раза);
# - `region` — регион. Имеет тип `object` и 688 уникальных значений. При этом наиболее часто встречающиеся значение `SF Bay Area` (встречается 1 750 раз);
# - `city` — город. Имеет тип `object` и 2 117 уникальных значений. При этом наиболее часто встречающиеся значение `San Francisco` (встречается 656 раз);
# - `funding_rounds` — количество раундов финансирования. Имеет тип `int64`, а значения распределены от 1 до 15 раундов. При этом среднее значение составляет примерно 1.71, а медиана — 1;
# - `first_funding_at` — дата первого раунда финансирования. Имеет тип `object` и 3 299 уникальных значений. При этом наиболее часто встречающиеся значение `2014-01-01` (встречается 126 раз);
# - `last_funding_at` — дата последнего раунда финансирования. Имеет тип `object` и 3 021 уникальных значения. При этом наиболее часто встречающиеся значение `2014-01-01` (встречается 111 раз);
# - `lifetime` — время существования стартапа в днях. Имеет тип `int64`, а значения распределены от 52 до 17 167 дней. При этом среднее значение составляет примерно 3097.61, а медиана — 2 526.

# ## Предобработка данных

# ### Проверка типов данных

# #### Тренировочный датафрейм

# Типы данных:

# In[24]:


startups_train.dtypes


# Изменение типов данных:

# In[25]:


# словарь для изменения типов данных
data_types_train = {
    'category_list': 'category',
    'status': 'category',
    'country_code': 'category',
    'state_code': 'category',
    'region': 'category',
    'city': 'category',
    'founded_at': 'datetime',
    'first_funding_at': 'datetime',
    'last_funding_at': 'datetime',
    'closed_at': 'datetime'
}

# изменение типов данных
startups_train = change_data_types(
    df=startups_train,
    column_types=data_types_train
)


# **Вывод:**
# 
# В тренировочном датафрейме столбцы `founded_at`, `first_funding_at`, `last_funding_at`, `closed_at` были переведены из `object` в тип даты и времени (`datetime64[ns]`). Также столбцы `category_list`, `status`, `country_code`, `state_code`, `region`, `city` были переведены из `object` в категориальный тип данных (`category`). Остальные же столбцы имеют корректные типы данных.

# #### Тестовый датафрейм

# Типы данных:

# In[26]:


startups_test.dtypes


# Изменение типов данных:

# In[27]:


# преобразование в float32
startups_test['funding_total_usd'] = startups_test['funding_total_usd'].astype('float32')

# преобразование в тип даты и времени
date_cols_te = ['first_funding_at', 'last_funding_at']
for col in date_cols_te:
    startups_test[col] = pd.to_datetime(startups_test[col], format='%Y-%m-%d')

# словарь для изменения типов данных
data_types_test = {
    'category_list': 'category',
    'country_code': 'category',
    'state_code': 'category',
    'region': 'category',
    'city': 'category',
    'first_funding_at': 'datetime',
    'last_funding_at': 'datetime'
}

# изменение типов данных
startups_test = change_data_types(
    df=startups_test,
    column_types=data_types_test
)


# **Вывод:**
# 
# В тестовом датафрейме столбцы `first_funding_at`, `last_funding_at` были переведены из `object` в тип даты и времени (`datetime64[ns]`). Также столбцы `category_list`, `country_code`, `state_code`, `region`, `city` были переведены из `object` в категориальный тип данных (`category`). Остальные же столбцы имеют корректные типы данных.

# ### Изучение пропущенных значений

# #### Тренировочный датафрейм

# Отображение пропущенных значений:

# In[28]:


msno.bar(startups_train, color='#597dbf');


# Удаление столбца c большим количеством пропущенных значений:

# In[29]:


startups_train = startups_train.drop(columns=['closed_at'])

# проверка
startups_train.columns


# Заполнение пропущенных значений:

# In[30]:


# числовые столбцы
num_cols = list(startups_train.select_dtypes(include=['number']).columns)

# категориальные столбцы
cat_cols = list(startups_train.select_dtypes(include=['category']).columns)

# текстовые столбцы
obj_cols = list(startups_train.select_dtypes(include=['object']).columns)

# удаление целевого признака, так как у него нет пропущенных значений,
# а также его нет в тестовой выборке
cat_cols.remove('status')

# заполнение пропусков в числовых и категориальных столбцах
startups_train[num_cols + cat_cols + obj_cols], fillna = fill_nan(
    startups_train[num_cols + cat_cols + obj_cols],
    num_cols=num_cols,
    cat_cols=cat_cols,
    obj_cols=obj_cols
)

# проверка
msno.bar(startups_train, color='#597dbf');


# **Вывод**
# 
# В тренировочном датафрейме числовой столбец `funding_total_usd` содержал 10 069 пропущенных значений, которые в последующем были заполнены медианным значением. А в столбцах `name` (1 пропуск), `category_list` (2 465 пропусков), `country_code` (5 502 пропуска), `state_code` (6 763 пропуска), `region` (6 359 пропусков), `city` (6 359 пропусков) пропущенные значения были заполнены значением `unknown` (неизвестно). Также последний столбец (`closed_at`) был удален, так как в нем было слишком много пропущенных значений (иначе говоря, он не нес никакой полезной информации для дальнейшего анализа).

# #### Тестовый датафрейм

# Отображение пропущенных значений:

# In[31]:


msno.bar(startups_test, color='#597dbf');


# Заполнение пропущенных значений:

# In[32]:


# заполнение пропусков
startups_test[num_cols + cat_cols + obj_cols], fillna = fill_nan(
    startups_test[num_cols + cat_cols + obj_cols],
    fillna_pipeline=fillna,
    num_cols=num_cols,
    cat_cols=cat_cols,
    obj_cols=obj_cols
)

# проверка
msno.bar(startups_test, color='#597dbf');


# **Вывод**
# 
# В тестовом датафрейме числовой столбец `funding_total_usd` содержал 2 578 пропущенных значений, которые в последующем были заполнены медианным значением из тренировочной выборки. А в столбцах `category_list` (591 пропуск), `country_code` (1 382 пропуска), `state_code` (1 695 пропусков), `region` (1 589 пропусков), `city` (1 587 пропусков) пропущенные значения были заполнены значением `unknown` (неизвестно).

# ### Изучение дубликатов

# #### Тренировочный датафрейм

# Уникальные значения столбцов типа `category`:

# In[33]:


cat_unique(startups_train)


# Укрупнение категорий:

# In[34]:


# кол-во уникальных значений до преобразований
before_tr = startups_train['category_list'].nunique()

# преобразование категорий
startups_train['category_list'] = startups_train['category_list'].apply(categorize)

# кол-во уникальных значений после преобразований
after_tr = startups_train['category_list'].nunique()

# вывод результатов
print(f'кол-во уникальных значений до: {before_tr}')
print(f'кол-во уникальных значений после: {after_tr}')


# Проверка кодов стран:

# In[35]:


# список всех кодов стран ISO 3166-1 alpha-3
valid_country_codes = [country.alpha_3 for country in pycountry.countries]

# добавление неизвестного значения в список правильных кодов стран
valid_country_codes.append('unknown')

# проверка
check_abbr(df=startups_train, col='country_code', valid=valid_country_codes)


# Проверка кодов штатов:

# In[36]:


# список всех кодов штатов
valid_state_codes = [state.abbr for state in us.states.STATES]

# добавление неизвестного значения в список правильных кодов штатов
valid_state_codes.append('unknown')

# проверка
check_abbr(df=startups_train, col='state_code', valid=valid_state_codes)


# Преобразование регистра в названиях регионов и городов:

# In[37]:


# в столбце 'region'
startups_train['region'] = startups_train['region'].str.title()
startups_train['region'] = startups_train['region'].str.replace('Unknown', 'unknown')

# в столбце 'city'
startups_train['city'] = startups_train['city'].str.title()
startups_train['city'] = startups_train['city'].str.replace('Unknown', 'unknown')

# проверка
print(f'region: {startups_train["region"].unique()}\n')
print(f'city: {startups_train["city"].unique()}')


# Кол-во явных дубликатов:

# In[38]:


# интересующие столбцы
comb_cols = list(startups_train.columns)
comb_cols.remove('name')

# вывод дубликатов
obvious_duplicates(startups_train, [comb_cols])


# Удаление явных дубликатов:

# In[39]:


# удаление дубликатов только по столбцу 'name'
startups_train = startups_train.drop_duplicates(subset=['name']).reset_index(drop=True)

# удаление дубликатов по всем столбцам, кроме 'name'
startups_train = startups_train.drop_duplicates(subset=comb_cols).reset_index(drop=True)

# проверка
obvious_duplicates(startups_train, [comb_cols])


# **Вывод:**
# 
# В тренировочном датафрейме было произведено укрупнение категорий (`category_list`) с 22 106 до 1 394 уникальных значений. А в столбцах `country_code` и `state_code` были замечены некорректные значения аббревиатур (40 неправильных кодов стран, 16 301 неправильных кодов штатов), которые в последующем были заменены на значение `unknown` (неизвестно). Также было удалено 24 явных дубликата по всем столбцам, кроме `name`.

# #### Тестовый датафрейм

# Уникальные значения столбцов типа `object`:

# In[40]:


cat_unique(startups_test)


# Укрупнение категорий:

# In[41]:


# кол-во уникальных значений до преобразований
before_te = startups_test['category_list'].nunique()

# преобразование категорий
startups_test['category_list'] = startups_test['category_list'].apply(categorize)

# кол-во уникальных значений после преобразований
after_te = startups_test['category_list'].nunique()

# вывод результатов
print(f'кол-во уникальных значений до: {before_te}')
print(f'кол-во уникальных значений после: {after_te}')


# Проверка кодов стран:

# In[42]:


check_abbr(df=startups_test, col='country_code', valid=valid_country_codes)


# Проверка кодов штатов:

# In[43]:


check_abbr(df=startups_test, col='state_code', valid=valid_state_codes)


# Преобразование регистра в названиях регионов и городов:

# In[44]:


# в столбце 'region'
startups_test['region'] = startups_test['region'].str.title()
startups_test['region'] = startups_test['region'].str.replace('Unknown', 'unknown')

# в столбце 'city'
startups_test['city'] = startups_test['city'].str.title()
startups_test['city'] = startups_test['city'].str.replace('Unknown', 'unknown')

# проверка
print(f'region: {startups_test["region"].unique()}\n')
print(f'city: {startups_test["city"].unique()}')


# Кол-во явных дубликатов:

# In[45]:


# удаление лишних столбцов из списка
comb_cols.remove('founded_at')
comb_cols.remove('status')

# отображение кол-ва явных дубликатов
obvious_duplicates(startups_test, [comb_cols])


# **Вывод:**
# 
# В тестовом датафрейме было произведено укрупнение категорий (`category_list`) с 6 207 до 666 уникальных значений. А в столбцах `country_code` и `state_code` были замечены некорректные значения аббревиатур (6 неправильных кодов стран, 4 060 неправильных кодов штатов), которые в последующем были заменены на значение `unknown` (неизвестно). Также было выявлено 59 явных дубликатов по всем столбцам, кроме `name`, но они не были удалены, поскольку нельзя вносить какие-либо координальные изменения в тестовую выборку.

# ### Изучение аномальных значений

# #### Тренировочный датафрейм

# Вывод расширенного статистического описания для всех числовых столбцов:

# In[46]:


for col in num_cols:
    result = more_describe(df=startups_train, col=col)
    print(f'Описание для столбца: {col}')
    print(result)
    print('\n')


# Построение ящиков с усами для каждого числового столбца:

# In[47]:


box_plots(startups_train, t='Ящики с усами для каждого числового столбца тренировочного датафрейма')


# Удаление аномального значения в столбце `funding_total_usd`:

# In[48]:


# удаление аномально большой общей суммы финансирования
startups_train = startups_train[startups_train['funding_total_usd'] <= 0.5e10]

# удаление аномально низкой общей суммы финансирования
startups_train = startups_train[startups_train['funding_total_usd'] >= 1_000]

# проверка
for col in num_cols:
    result = more_describe(df=startups_train, col=col)
    print(f'Описание для столбца: {col}')
    print(result)
    print('\n')
    
box_plots(startups_train, t='Ящики с усами для каждого числового столбца тренировочного датафрейма')


# **Вывод:**
# 
# В столбце `funding_total_usd` тренировочного датафрейма было удалено 55 аномалий, поскольку они имели слишком большие (больше 5 млрд. долларов) и малые значения (меньше 1 000 долларов).

# #### Тестовый датафрейм

# Вывод расширенного статистического описания для всех числовых столбцов:

# In[49]:


for col in (num_cols + ['lifetime']):
    result = more_describe(df=startups_test, col=col)
    print(f'Описание для столбца: {col}')
    print(result)
    print('\n')


# Построение ящиков с усами для каждого числового столбца:

# In[50]:


box_plots(startups_test, t='Ящики с усами для каждого числового столбца тестового датафрейма')


# **Вывод:**
# 
# В столбце `funding_total_usd` тестового датафрейма тоже наблюдаются аномально низкие значения, но они не были удалены, поскольку нельзя вносить какие-либо координальные изменения в тестовую выборку.

# ### Удаление неинформативных признаков

# Вывод всех столбцов из тренировочного и тестового датафреймов:

# In[51]:


# список всех столбцов тренировочного датафрейма
print(f'Столбцы тренировочного датафрейма:\n {list(startups_train.columns)}\n')

# список всех столбцов тестового датафрейма
print(f'Столбцы тестового датафрейма:\n {list(startups_test.columns)}')


# Удаление лишних столбцов:

# In[52]:


# удаление даты основания стартапа из тренировочного датафрейма
startups_train = startups_train.drop('founded_at', axis=1)

# удаление времени существования стартапа в днях из тестового датафрейма
startups_test = startups_test.drop('lifetime', axis=1)

# проверка
print(f'Столбцы тренировочного датафрейма:\n {list(startups_train.columns)}\n')
print(f'Столбцы тестового датафрейма:\n {list(startups_test.columns)}')


# **Вывод:**
# 
# В тренировочном датафрейме была удалена дата основания стартапа (`founded_at`), так как такого признака нет в тестовой выборке. Также в тестовом датафрейме было удалено время существования стартапа в днях (`lifetime`), поскольку такого признака нет в тренировочной выборке, а воссоздать его не представляется возможным.

# ### Вывод предобработки данных
# 
# #### Проверка типов данных
# 
# В тренировочном датафрейме столбцы `founded_at`, `first_funding_at`, `last_funding_at`, `closed_at` были переведены из `object` в тип даты и времени (`datetime64[ns]`). Также столбцы `category_list`, `status`, `country_code`, `state_code`, `region`, `city` были переведены из `object` в категориальный тип данных (`category`). Остальные же столбцы имеют корректные типы данных.
# 
# В тестовом датафрейме столбцы `first_funding_at`, `last_funding_at` были переведены из `object` в тип даты и времени (`datetime64[ns]`). Также столбцы `category_list`, `country_code`, `state_code`, `region`, `city` были переведены из `object` в категориальный тип данных (`category`). Остальные же столбцы имеют корректные типы данных.
# 
# #### Изучение пропущенных значений
# 
# В тренировочном датафрейме числовой столбец `funding_total_usd` содержал 10 069 пропущенных значений, которые в последующем были заполнены медианным значением. А в столбцах `name` (1 пропуск), `category_list` (2 465 пропусков), `country_code` (5 502 пропуска), `state_code` (6 763 пропуска), `region` (6 359 пропусков), `city` (6 359 пропусков) пропущенные значения были заполнены значением `unknown` (неизвестно). Также последний столбец (`closed_at`) был удален, так как в нем было слишком много пропущенных значений (иначе говоря, он не нес никакой полезной информации для дальнейшего анализа).
# 
# В тестовом датафрейме числовой столбец `funding_total_usd` содержал 2 578 пропущенных значений, которые в последующем были заполнены медианным значением из тренировочной выборки. А в столбцах `category_list` (591 пропуск), `country_code` (1 382 пропуска), `state_code` (1 695 пропусков), `region` (1 589 пропусков), `city` (1 587 пропусков) пропущенные значения были заполнены значением `unknown` (неизвестно).
# 
# #### Изучение дубликатов
# 
# В тренировочном датафрейме было произведено укрупнение категорий (`category_list`) с 22 106 до 1 394 уникальных значений. А в столбцах `country_code` и `state_code` были замечены некорректные значения аббревиатур (40 неправильных кодов стран, 16 301 неправильных кодов штатов), которые в последующем были заменены на значение `unknown` (неизвестно). Также было удалено 24 явных дубликатов по всем столбцам, кроме `name`.
# 
# В тестовом датафрейме было произведено укрупнение категорий (`category_list`) с 6 207 до 666 уникальных значений. А в столбцах `country_code` и `state_code` были замечены некорректные значения аббревиатур (6 неправильных кодов стран, 4 060 неправильных кодов штатов), которые в последующем были заменены на значение `unknown` (неизвестно). Также было выявлено 59 явных дубликатов по всем столбцам, кроме `name`, но они не были удалены, поскольку нельзя вносить какие-либо координальные изменения в тестовую выборку.
# 
# #### Изучение аномальных значений
# 
# В столбце `funding_total_usd` тренировочного датафрейма было удалено 55 аномалий, поскольку они имели слишком большие (больше 5 млрд. долларов) и малые значения (меньше 1 000 долларов).
# 
# В столбце `funding_total_usd` тестового датафрейма тоже наблюдаются аномально низкие значения, но они не были удалены, поскольку нельзя вносить какие-либо координальные изменения в тестовую выборку.
# 
# #### Удаление неинформативных признаков
# 
# В тренировочном датафрейме была удалена дата основания стартапа (`founded_at`), так как такого признака нет в тестовой выборке. Также в тестовом датафрейме было удалено время существования стартапа в днях (`lifetime`), поскольку такого признака нет в тренировочной выборке, а воссоздать его не представляется возможным.

# ## Исследовательский анализ данных

# ### Распределение категорий стартапов

# #### Тренировочная выборка

# In[53]:


tree_map(
    df=startups_train,
    cat='category_list',
    subcat='status',
    t='Распределение категорий стартапов в тренировочной выборке'
)


# **Вывод:**
# 
# В тренировочной выборке большая часть стартапов имеет категорию `IT`, их доля составляет 17.67% (8 561 действующих, 701 закрытых). На втором месте находится категория `Medicine` с долей в 11.65% (5 770 действующих, 343 закрытых), на третьем — `Media` c долей в 8.47% (3 810 действующих, 629 закрытых), на четвертом — `Other` с долей в 5.46% (2 675 действующих, 188 закрытых), на пятом — `unknown` с долей в 4.67% (1 727 действующих, 726 закрытых).

# #### Тестовая выборка

# In[54]:


tree_map(
    df=startups_test,
    cat='category_list',
    t='Распределение категорий стартапов в тестовой выборке'
)


# **Вывод:**
# 
# В тестовой выборке большая часть стартапов имеет категорию `IT`, их доля составляет 17.45% (2 290 стартапов). На втором месте находится категория `Medicine` с долей в 11.49% (1 508 стартапов), на третьем — `Media` c долей в 8.25% (1 083 стартапа), на четвертом — `Other` с долей в 5.61% (736 стартапов), на пятом — `unknown` с долей в 4.5% (591 стартап).

# ### Распределение общих сумм финансирования стартапов

# #### Тренировочная выборка

# Вывод расширенного статистического описания:

# In[55]:


more_describe(df=startups_train, col='funding_total_usd', hue='status')


# Составление графика:

# In[56]:


hist_plot(
    df=startups_train,
    x='funding_total_usd',
    hue='status',
    xlim=(0, 16.64835e6),
    xl='общая сумма финансирования (в долларах)',
    yl='кол-во стартапов',
    lt='Статусы',
    t='Распределение общих сумм финансирования стартапов в тренировочной выборке'
)


# **Вывод:**
# 
# Распределение общих сумм финансирования стартапов в тренировочной выборке имеет сильную правостороннюю асимметрию, при этом пик у действующих стартапов находится в диапазоне значений 1-3 млн. долларов (16 034 стартапов), а у закрывшихся — 0.001-1 млн. долларов (1 716 стартапов). Также у обоих распределений минимальные значения совпадают (в значении 1 тыс. долларов), а максимальные расходятся (у действующих стартапов — примерно 4.812 млрд. долларов, а у закрывшихся — 1.567 млрд. долларов). В предоставленных данных, обе группы стартапов имеют одинаковые общие суммы финансирования, так как их меиданы совпадают в значении 2 млн. долларов. При этом межквартильный размах у действующих стартапов шире (примерно на 1.81 млн. долларов), что указывает на более широкое распределение данных. Также у действующих стартапов наблюдается 15.06% выбросов (в значениях больше 16.648 млн. долларов), а у закрывшихся — 14.73% (в значениях больше 11.885 млн. долларов).

# #### Тестовая выборка

# Вывод расширенного статистического описания:

# In[57]:


more_describe(df=startups_test, col='funding_total_usd')


# Составление графика:

# In[58]:


hist_plot(
    df=startups_test,
    x='funding_total_usd',
    xlim=(0, 15.5e6),
    xl='общая сумма финансирования (в долларах)',
    yl='кол-во стартапов',
    t='Распределение общих сумм финансирования стартапов в тестовой выборке'
)


# **Вывод:**
# 
# Распределение общих сумм финансирования стартапов в тестовой выборке имеет сильную правостороннюю асимметрию, при этом пик находится в диапазоне значений от 1 доллара до 2.5 млн. долларов (8 245 стартапов). Также минимальное значение составляет 1 доллар, а максимальное — 4.715 млрд. долларов. Межквартильный размах составляет примерно 6.013 млн. долларов. Обнаружено 14.83% выбросов (в значениях больше 15.5 млн. долларов).

# ### Распределение географических местоположений стартапов

# #### Тренировочная выборка

# In[59]:


# графическое отображение всех стран с их регионами и городами
tree_map(
    df=startups_train,
    cat='country_code',
    subcat='region',
    subsubcat='city',
    t='Распределение географических местоположений стартапов в тренировочной выборке'
)

# графическое отображение всех стран и городов с разбивкой по статусу стартапов
tree_map(
    df=startups_train,
    cat='country_code',
    subcat='city',
    subsubcat='status',
    t='Распределение географических местоположений стартапов с разбивкой по их статусу в тренировочной выборке'
)

# графическое отображение всех штатов с разбивкой по статусу стартапов
tree_map(
    df=startups_train,
    cat='state_code',
    subcat='status',
    t='Распределение штатов с разбивкой по статусу стартапов в тренировочной выборке'
)


# **Вывод:**
# 
# В тренировочной выборке большая часть стартапов расположена в `USA` в наиболее популярном регионе `SF Bay Area` (города: `San Francisco` — 4.89% действующих и 0.49% закрывшихся стартапов, `Palo Alto` — 1.03% действующих и 0.12% закрывшихся стартапов, `Mountain View` — 0.84% действующих и 0.05% закрывшихся стартапов), при этом наиболее популярные штаты: `unknown` (38.87% действующих и 5.04% закрывшихся стартапов), `CA` (17.69% действующих и 1.78% закрывшихся стартапов), `NY` (5.47% действующих и 0.46% закрывшихся стартапов). На втором месте расположились неизвестные страны с неизвестными регионами и городами (8.01% действующих и 2.44% закрывшихся стартапов). На третьем — `GBR` с наиболее популярным регионом `London` (города: `London` — 2.67% действующих и 0.2% закрывшихся стартапов, `Cambridge` — 0.18% действующих и 0.01% закрывшихся стартапов, `Oxford` — 0.06% действующих и 0.002% закрывшихся стартапов). На четвертом — `CAN` с наиболее популярным регионом `Toronto` (города: `Toronto` — 0.73% действующих и 0.08% закрывшихся стартапов, `Waterloo` — 0.06% действующих и 0.01% закрывшихся стартапов, `Kitchener` — 0.05% действующих и 0.002% закрывшихся стартапов). На пятом — `FRA` с наиболее популярным регионом `Paris` (города: `Paris` — 0.84% действующих и 0.08% закрывшихся стартапов, `Boulogne-Billancourt` — 0.03% действующих и 0.002% закрывшихся стартапов, `Courbevoie` — 0.01% действующих и 0% закрывшихся стартапов).

# #### Тестовая выборка

# In[60]:


# графическое отображение всех стран с их регионами и городами
tree_map(
    df=startups_test,
    cat='country_code',
    subcat='region',
    subsubcat='city',
    t='Распределение географических местоположений стартапов в тестовой выборке'
)

# графическое отображение всех штатов с разбивкой по статусу стартапов
tree_map(
    df=startups_test,
    cat='state_code',
    t='Распределение штатов стартапов в тестовой выборке'
)


# **Вывод:**
# 
# В тестовой выборке большая часть стартапов расположена в `USA` в наиболее популярном регионе `SF Bay Area` (города: `San Francisco` — 5% стартапов, `Palo Alto` — 1.18% стартапов, `Mountain View` — 0.99% стартапов), при этом наиболее популярные штаты: `unknown` (43.85% стартапов), `CA` (19.44% стартапов), `NY` (6.01% стартапов). На втором месте расположились неизвестные страны с неизвестными регионами и городами (10.54% стартапов). На третьем — `GBR` с наиболее популярным регионом `London` (города: `London` — 2.91% стартапов, `Cambridge` — 0.22% стартапов, `Oxford` — 0.06% стартапов). На четвертом — `CAN` с наиболее популярным регионом `Toronto` (города: `Toronto` — 0.72% стартапов, `Waterloo` — 0.11% стартапов, `Kitchener` — 0.08% стартапов). На пятом — `DEU` с наиболее популярным регионом `Berlin` (города: `Berlin` — 0.36% стартапов, `Potsdam` — 0.03% стартапов, `Berlin-Baumschulenweg` — 0.01% стартапов).

# ### Распределение кол-ва раундов финансирования стартапов

# #### Тренировочная выборка

# Вывод расширенного статистического описания:

# In[61]:


more_describe(df=startups_train, col='funding_rounds', hue='status')


# Составление графика:

# In[62]:


hist_plot(
    df=startups_train,
    x='funding_rounds',
    hue='status',
    xl='кол-во раундов',
    yl='кол-во стартапов',
    lt='Статусы',
    t='Распределение кол-ва раундов финансирования стартапов в тренировочной выборке'
)


# **Вывод:**
# 
# Распределение кол-ва раундов финансирования стартапов в тренировочной выборке имеет сильную правостороннюю асимметрию, при этом пик у действующих (29 666 стартапов) и закрывшихся (3 751 стартап) стартапов находится в значении 1-го раунда. Также у обоих распределений минимальные значения совпадают (в значении 1-го раунда), а максимальные расходятся (у действующих стартапов — 19 раундов, а у закрывшихся — 11 раундов). В предоставленных данных, обе группы стартапов имеют одинаковые кол-ва раундов финансирования, так как их меиданы совпадают в значении 1-го раунда. При этом межквартильный размах у действующих стартапов шире (на один раунд), что указывает на более широкое распределение данных. Также у действующих стартапов наблюдается 9.59% выбросов (в значениях больше 3-х раундов), а у закрывшихся — 23.59% (в значениях больше 1-го раунда).

# #### Тестовая выборка

# Вывод расширенного статистического описания:

# In[63]:


more_describe(df=startups_test, col='funding_rounds')


# Составление графика:

# In[64]:


hist_plot(
    df=startups_test,
    x='funding_rounds',
    xl='кол-во раундов',
    yl='кол-во стартапов',
    t='Распределение кол-ва раундов финансирования стартапов в тестовой выборке'
)


# **Вывод:**
# 
# Распределение кол-ва раундов финансирования стартапов в тестовой выборке имеет сильную правостороннюю асимметрию, при этом пик находится в значении 1-го раунда (8 449 стартапов). Также минимальное значение составляет 1 раунд, а максимальное — 15 раундов. Межквартильный размах составляет 1 раунд. Обнаружено 8.68% выбросов (в значениях больше 3 раундов).

# ### Распределение дат первого раунда финансирования стартапов

# #### Тренировочная выборка

# Вывод расширенного статистического описания:

# In[65]:


more_describe(df=startups_train, col='first_funding_at', hue='status')


# Составление графика:

# In[66]:


hist_plot(
    df=startups_train,
    x='first_funding_at',
    hue='status',
    xl='дата первого раунда финансирования',
    yl='кол-во стартапов',
    lt='Статусы',
    t='Распределение дат первого раунда финансирования стартапов в тренировочной выборке'
)


# **Вывод:**
# 
# Распределение дат первого раунда финансирования стартапов в тренировочной выборке имеет сильную левостороннюю асимметрию, при этом пик у действующих стартапов находится в диапазоне значений с января по февраль 2014 года (1 746 стартапов), а у закрывшихся — с мая по инюнь 2015 года (172 стартапа). Также у обоих распределений минимальные (у действующих стартапов — `1977-05-15`, у закрывшихся стартапов — `1982-03-20`) и максимальные (у действующих стартапов — `2015-12-05`, у закрывшихся стартапов — `2015-12-04`) значения не совпадают. В предоставленных данных, обе группы стартапов имеют разные даты первого раунда финансирования, так как их меиданы не совпадают (у действующих стартапов — `2012-09-13 12:00:00`, у закрывшихся стартапов — `2010-08-10`). При этом межквартильный размах у закрывшихся стартапов шире (на 566 дней), что указывает на более широкое распределение данных. Также у действующих стартапов наблюдается 3.58% выбросов (в значениях до августа 2003), а у закрывшихся — 0.94% (в значениях до января 1999).

# #### Тестовая выборка

# Вывод расширенного статистического описания:

# In[67]:


more_describe(df=startups_test, col='first_funding_at')


# Составление графика:

# In[68]:


hist_plot(
    df=startups_test,
    x='first_funding_at',
    xl='дата первого раунда финансирования',
    yl='кол-во стартапов',
    t='Распределение дат первого раунда финансирования стартапов в тестовой выборке'
)


# **Вывод:**
# 
# Распределение дат первого раунда финансирования стартапов в тестовой выборке имеет сильную правостороннюю асимметрию, при этом пик находится в диапазоне значений с января по февраль 2014 года (469 стартапов). Также минимальным значением является `1973-04-15`, а максимальным — `2015-12-03`. Межквартильный размах составляет 1 597 дней. Обнаружено 3.75% выбросов (в значениях до апреля 2003).

# ### Распределение дат последнего раунда финансирования стартапов

# #### Тренировочная выборка

# Вывод расширенного статистического описания:

# In[69]:


more_describe(df=startups_train, col='last_funding_at', hue='status')


# Составление графика:

# In[70]:


hist_plot(
    df=startups_train,
    x='last_funding_at',
    hue='status',
    xl='дата последнего раунда финансирования',
    yl='кол-во стартапов',
    lt='Статусы',
    t='Распределение дат последнего раунда финансирования стартапов в тренировочной выборке'
)


# **Вывод:**
# 
# Распределение дат последнего раунда финансирования стартапов в тренировочной выборке имеет сильную левостороннюю асимметрию, при этом пик у действующих стартапов находится в диапазоне значений с марта по апрель 2015 года (2 203 стартапа), а у закрывшихся — с мая по инюнь 2015 года (188 стартапов). Также у обоих распределений минимальные (у действующих стартапов — `1977-05-15`, у закрывшихся стартапов — `1982-03-20`) и максимальные (у действующих стартапов — `2015-12-07`, у закрывшихся стартапов — `2015-12-04`) значения не совпадают. В предоставленных данных, обе группы стартапов имеют разные даты последнего раунда финансирования, так как их меиданы не совпадают (у действующих стартапов — `2013-10-15`, у закрывшихся стартапов — `2011-04-01`). При этом межквартильный размах у закрывшихся стартапов шире (на 773 дня), что указывает на более широкое распределение данных. Также у действующих стартапов наблюдается 5.03% выбросов (в значениях до марта 2006), а у закрывшихся — 0.94% (в значениях до ноября 1999).

# #### Тестовая выборка

# Вывод расширенного статистического описания:

# In[71]:


more_describe(df=startups_test, col='last_funding_at')


# Составление графика:

# In[72]:


hist_plot(
    df=startups_test,
    x='last_funding_at',
    xl='дата последнего раунда финансирования',
    yl='кол-во стартапов',
    t='Распределение дат последнего раунда финансирования стартапов в тестовой выборке'
)


# **Вывод:**
# 
# Распределение дат последнего раунда финансирования стартапов в тестовой выборке имеет сильную правостороннюю асимметрию, при этом пик находится в диапазоне значений с марта по апрель 2015 года (575 стартапов). Также минимальным значением является `1973-04-15`, а максимальным — `2015-12-12`. Межквартильный размах составляет 1 401 день. Обнаружено 4.36% выбросов (в значениях до апреля 2005).

# ### Распределение статусов стартапов в тренировочной выборке

# In[73]:


pie_plot(df=startups_train,
         col='status',
         lt='Статусы',
         t='Распределение статусов стартапов в тренировочной выборке'
)


# **Вывод:**
# 
# В тренировочной выборке наблюдается огромный дисбаланс категорий целевого признака `status` в сторону действующих стартапов, имеющих долю в 90.6% (47 528 стартапов). При этом доля у закрывшихся стартапов составляет 9.36% (4 909 стартапов).

# ### Вывод исследовательского анализа данных
# 
# #### Распределение категорий стартапов
# 
# В тренировочной выборке большая часть стартапов имеет категорию `IT`, их доля составляет 17.67% (8 561 действующих, 701 закрытых). На втором месте находится категория `Medicine` с долей в 11.65% (5 770 действующих, 343 закрытых), на третьем — `Media` c долей в 8.47% (3 810 действующих, 629 закрытых), на четвертом — `Other` с долей в 5.46% (2 675 действующих, 188 закрытых), на пятом — `unknown` с долей в 4.67% (1 727 действующих, 726 закрытых).
# 
# В тестовой выборке большая часть стартапов имеет категорию `IT`, их доля составляет 17.45% (2 290 стартапов). На втором месте находится категория `Medicine` с долей в 11.49% (1 508 стартапов), на третьем — `Media` c долей в 8.25% (1 083 стартапа), на четвертом — `Other` с долей в 5.61% (736 стартапов), на пятом — `unknown` с долей в 4.5% (591 стартап).
# 
# #### Распределение общих сумм финансирования стартапов
# 
# Распределение общих сумм финансирования стартапов в тренировочной выборке имеет сильную правостороннюю асимметрию, при этом пик у действующих стартапов находится в диапазоне значений 1-3 млн. долларов (16 034 стартапов), а у закрывшихся — 0.001-1 млн. долларов (1 716 стартапов). Также у обоих распределений минимальные значения совпадают (в значении 1 тыс. долларов), а максимальные расходятся (у действующих стартапов — примерно 4.812 млрд. долларов, а у закрывшихся — 1.567 млрд. долларов). В предоставленных данных, обе группы стартапов имеют одинаковые общие суммы финансирования, так как их меиданы совпадают в значении 2 млн. долларов. При этом межквартильный размах у действующих стартапов шире (примерно на 1.81 млн. долларов), что указывает на более широкое распределение данных. Также у действующих стартапов наблюдается 15.06% выбросов (в значениях больше 16.648 млн. долларов), а у закрывшихся — 14.73% (в значениях больше 11.885 млн. долларов).
# 
# Распределение общих сумм финансирования стартапов в тестовой выборке имеет сильную правостороннюю асимметрию, при этом пик находится в диапазоне значений от 1 доллара до 2.5 млн. долларов (8 245 стартапов). Также минимальное значение составляет 1 доллар, а максимальное — 4.715 млрд. долларов. Межквартильный размах составляет примерно 6.013 млн. долларов. Обнаружено 14.83% выбросов (в значениях больше 15.5 млн. долларов).
# 
# #### Распределение географических местоположений стартапов
# 
# В тренировочной выборке большая часть стартапов расположена в `USA` в наиболее популярном регионе `SF Bay Area` (города: `San Francisco` — 4.89% действующих и 0.49% закрывшихся стартапов, `Palo Alto` — 1.03% действующих и 0.12% закрывшихся стартапов, `Mountain View` — 0.84% действующих и 0.05% закрывшихся стартапов), при этом наиболее популярные штаты: `unknown` (38.87% действующих и 5.04% закрывшихся стартапов), `CA` (17.69% действующих и 1.78% закрывшихся стартапов), `NY` (5.47% действующих и 0.46% закрывшихся стартапов). На втором месте расположились неизвестные страны с неизвестными регионами и городами (8.01% действующих и 2.44% закрывшихся стартапов). На третьем — `GBR` с наиболее популярным регионом `London` (города: `London` — 2.67% действующих и 0.2% закрывшихся стартапов, `Cambridge` — 0.18% действующих и 0.01% закрывшихся стартапов, `Oxford` — 0.06% действующих и 0.002% закрывшихся стартапов). На четвертом — `CAN` с наиболее популярным регионом `Toronto` (города: `Toronto` — 0.73% действующих и 0.08% закрывшихся стартапов, `Waterloo` — 0.06% действующих и 0.01% закрывшихся стартапов, `Kitchener` — 0.05% действующих и 0.002% закрывшихся стартапов). На пятом — `FRA` с наиболее популярным регионом `Paris` (города: `Paris` — 0.84% действующих и 0.08% закрывшихся стартапов, `Boulogne-Billancourt` — 0.03% действующих и 0.002% закрывшихся стартапов, `Courbevoie` — 0.01% действующих и 0% закрывшихся стартапов).
# 
# В тестовой выборке большая часть стартапов расположена в `USA` в наиболее популярном регионе `SF Bay Area` (города: `San Francisco` — 5% стартапов, `Palo Alto` — 1.18% стартапов, `Mountain View` — 0.99% стартапов), при этом наиболее популярные штаты: `unknown` (43.85% стартапов), `CA` (19.44% стартапов), `NY` (6.01% стартапов). На втором месте расположились неизвестные страны с неизвестными регионами и городами (10.54% стартапов). На третьем — `GBR` с наиболее популярным регионом `London` (города: `London` — 2.91% стартапов, `Cambridge` — 0.22% стартапов, `Oxford` — 0.06% стартапов). На четвертом — `CAN` с наиболее популярным регионом `Toronto` (города: `Toronto` — 0.72% стартапов, `Waterloo` — 0.11% стартапов, `Kitchener` — 0.08% стартапов). На пятом — `DEU` с наиболее популярным регионом `Berlin` (города: `Berlin` — 0.36% стартапов, `Potsdam` — 0.03% стартапов, `Berlin-Baumschulenweg` — 0.01% стартапов).
# 
# #### Распределение дат первого раунда финансирования стартапов
# 
# Распределение дат первого раунда финансирования стартапов в тренировочной выборке имеет сильную левостороннюю асимметрию, при этом пик у действующих стартапов находится в диапазоне значений с января по февраль 2014 года (1 746 стартапов), а у закрывшихся — с мая по инюнь 2015 года (172 стартапа). Также у обоих распределений минимальные (у действующих стартапов — `1977-05-15`, у закрывшихся стартапов — `1982-03-20`) и максимальные (у действующих стартапов — `2015-12-05`, у закрывшихся стартапов — `2015-12-04`) значения не совпадают. В предоставленных данных, обе группы стартапов имеют разные даты первого раунда финансирования, так как их меиданы не совпадают (у действующих стартапов — `2012-09-13 12:00:00`, у закрывшихся стартапов — `2010-08-10`). При этом межквартильный размах у закрывшихся стартапов шире (на 566 дней), что указывает на более широкое распределение данных. Также у действующих стартапов наблюдается 3.58% выбросов (в значениях до августа 2003), а у закрывшихся — 0.94% (в значениях до января 1999).
# 
# Распределение дат первого раунда финансирования стартапов в тестовой выборке имеет сильную правостороннюю асимметрию, при этом пик находится в диапазоне значений с января по февраль 2014 года (469 стартапов). Также минимальным значением является `1973-04-15`, а максимальным — `2015-12-03`. Межквартильный размах составляет 1 597 дней. Обнаружено 3.75% выбросов (в значениях до апреля 2003).
# 
# #### Распределение дат последнего раунда финансирования стартапов
# 
# Распределение дат последнего раунда финансирования стартапов в тренировочной выборке имеет сильную левостороннюю асимметрию, при этом пик у действующих стартапов находится в диапазоне значений с марта по апрель 2015 года (2 203 стартапа), а у закрывшихся — с мая по инюнь 2015 года (188 стартапов). Также у обоих распределений минимальные (у действующих стартапов — `1977-05-15`, у закрывшихся стартапов — `1982-03-20`) и максимальные (у действующих стартапов — `2015-12-07`, у закрывшихся стартапов — `2015-12-04`) значения не совпадают. В предоставленных данных, обе группы стартапов имеют разные даты последнего раунда финансирования, так как их меиданы не совпадают (у действующих стартапов — `2013-10-15`, у закрывшихся стартапов — `2011-04-01`). При этом межквартильный размах у закрывшихся стартапов шире (на 773 дня), что указывает на более широкое распределение данных. Также у действующих стартапов наблюдается 5.03% выбросов (в значениях до марта 2006), а у закрывшихся — 0.94% (в значениях до ноября 1999).
# 
# Распределение дат последнего раунда финансирования стартапов в тестовой выборке имеет сильную правостороннюю асимметрию, при этом пик находится в диапазоне значений с марта по апрель 2015 года (575 стартапов). Также минимальным значением является `1973-04-15`, а максимальным — `2015-12-12`. Межквартильный размах составляет 1 401 день. Обнаружено 4.36% выбросов (в значениях до апреля 2005).
# 
# #### Распределение статусов стартапов в тренировочной выборке
# 
# В тренировочной выборке наблюдается огромный дисбаланс категорий целевого признака `status` в сторону действующих стартапов, имеющих долю в 90.6% (47 528 стартапов). При этом доля у закрывшихся стартапов составляет 9.36% (4 909 стартапов).

# ## Разработка новых синтетических признаков

# ### Преобразование дат в числовой формат

# #### Тренировочная выборка

# In[74]:


startups_train['first_funding_at'] = pd.to_datetime(startups_train['first_funding_at']).astype('int64') // 10**9
startups_train['last_funding_at'] = pd.to_datetime(startups_train['last_funding_at']).astype('int64') // 10**9

# проверка
startups_train.dtypes


# **Вывод:**
# 
# В тренировочной выборке все даты были переведены в числовой формат (секунды от эпохи Unix).

# #### Тестовая выборка

# In[75]:


startups_test['first_funding_at'] = pd.to_datetime(startups_test['first_funding_at']).astype('int64') // 10**9
startups_test['last_funding_at'] = pd.to_datetime(startups_test['last_funding_at']).astype('int64') // 10**9

# проверка
startups_test.dtypes


# **Вывод:**
# 
# В тестовой выборке все даты были переведены в числовой формат (секунды от эпохи Unix).

# ### Получение координат городов стартапов

# #### Тренировочная выборка

# Получение координат всех городов:

# In[76]:


# get_coords_city(
    # df=startups_train,
    # col='city',
    # path_to_save='datasets/startups_train_with_coords.csv'
# )


# Считывание полученных координат:

# In[77]:


# перезапись датафрейма
startups_train = create_dataframe('datasets/startups_train_with_coords.csv')

# проверка
startups_train.head(10)


# Отображение типов данных:

# In[78]:


startups_train.dtypes


# Изменение типов данных:

# In[79]:


# словарь для изменения типов данных
data_types_train = {
    'category_list': 'category',
    'status': 'category',
    'country_code': 'category',
    'state_code': 'category',
    'region': 'category',
    'city': 'category',
    'latitude': 'float32',
    'longitude': 'float32'
}

# изменение типов данных
startups_train = change_data_types(
    df=startups_train,
    column_types=data_types_train
)


# Отображение пропущенных значений:

# In[80]:


msno.bar(startups_train, color='#597dbf');


# Заполнение пропущенных координат:

# In[81]:


startups_train[['latitude', 'longitude']] = startups_train[['latitude', 'longitude']].fillna({'latitude': 0.0, 'longitude': 0.0})

# проверка
msno.bar(startups_train, color='#597dbf');


# **Вывод:**
# 
# В тренировочной выборке созданы новые признаки, отображающие координаты городов (`latitude` — широта, `longitude` — долгота), в которых находятся стартапы. Причем их типы данных были переведны из `float64` в `float32` для экономии вычислительной памяти.

# #### Тестовая выборка

# Получение координат всех городов:

# In[82]:


# get_coords_city(
    # df=startups_test,
    # col='city',
    # path_to_save='datasets/startups_test_with_coords.csv'
# )


# Считывание полученных координат:

# In[83]:


# перезапись датафрейма
startups_test = create_dataframe('datasets/startups_test_with_coords.csv')

# проверка
startups_test.head(10)


# Отображение типов данных:

# In[84]:


startups_test.dtypes


# Изменение типов данных:

# In[85]:


# словарь для изменения типов данных
data_types_test = {
    'category_list': 'category',
    'country_code': 'category',
    'state_code': 'category',
    'region': 'category',
    'city': 'category',
    'latitude': 'float32',
    'longitude': 'float32'
}

# изменение типов данных
startups_test = change_data_types(
    df=startups_test,
    column_types=data_types_test
)


# Отображение пропущенных значений:

# In[86]:


msno.bar(startups_test, color='#597dbf');


# Заполнение пропущенных координат:

# In[87]:


startups_test[['latitude', 'longitude']] = startups_test[['latitude', 'longitude']].fillna({'latitude': 0.0, 'longitude': 0.0})

# проверка
msno.bar(startups_test, color='#597dbf');


# **Вывод:**
# 
# В тестовой выборке созданы новые признаки, отображающие координаты городов (`latitude` — широта, `longitude` — долгота), в которых находятся стартапы. Причем их типы данных были переведны из `float64` в `float32` для экономии вычислительной памяти.

# ### Векторизация категорий стартапов

# #### Тренировочная выборка

# In[88]:


# создание TF-IDF векторизатора
vectorizer = TfidfVectorizer(max_features=100)
tfidf_matrix_train = vectorizer.fit_transform(startups_train['category_list'])

# преобразование в DataFrame
tfidf_df_train = pd.DataFrame(tfidf_matrix_train.toarray(), columns=vectorizer.get_feature_names_out())

# использование PCA для уменьшения размерности до 10 признаков
pca = PCA(n_components=10)
tfidf_pca_train = pca.fit_transform(tfidf_df_train)

# преобразование в DataFrame
tfidf_pca_df_train = pd.DataFrame(tfidf_pca_train, columns=[f'category_pca_{i}' for i in range(10)])

# добавление в датафрейм
startups_train = pd.concat([startups_train, tfidf_pca_df_train], axis=1)

# проверка
startups_train.head(10)


# **Вывод:**
# 
# В тренировочной выборке была проведена векторизация категорий с помощью TF-IDF (максимальное кол-во признаков 100) и PCA (сокращение до 10 признаков).

# #### Тестовая выборка

# In[89]:


# преобразование категорий в TF-IDF с уже обученным vectorizer
tfidf_matrix_test = vectorizer.transform(startups_test['category_list'])

# преобразование в DataFrame
tfidf_df_test = pd.DataFrame(tfidf_matrix_test.toarray(), columns=vectorizer.get_feature_names_out())

# уменьшение размерности с уже обученным PCA
tfidf_pca_test = pca.transform(tfidf_df_test)

# преобразование в DataFrame
tfidf_pca_df_test = pd.DataFrame(tfidf_pca_test, columns=[f'category_pca_{i}' for i in range(10)])

# добавление в датафрейм
startups_test = pd.concat([startups_test, tfidf_pca_df_test], axis=1)

# Проверяем
startups_test.head(10)


# **Вывод:**
# 
# В тестовой выборке была проведена векторизация категорий с помощью обученных на тренировочной выборке TF-IDF (максимальное кол-во признаков 100) и PCA (сокращение до 10 признаков).

# ### Вывод разработки новых синтетических признаков
# 
# #### Преобразование дат в числовой формат
# 
# В тренировочной и тестовой выборках все даты были переведены в числовой формат (секунды от эпохи Unix).
# 
# #### Получение координат городов стартапов
# 
# В тренировочной и тестовой выборках созданы новые признаки, отображающие координаты городов (`latitude` — широта, `longitude` — долгота), в которых находятся стартапы. Причем их типы данных были переведны из `float64` в `float32` для экономии вычислительной памяти.
# 
# #### Векторизация категорий стартапов
# 
# В тренировочной и тестовой выборках была проведена векторизация категорий с помощью TF-IDF (максимальное кол-во признаков 100) и PCA (сокращение до 10 признаков).

# ## Проверка на мультиколлинеарность

# Построение тепловой карты:

# In[90]:


# числовые столбцы
num_cols = list(startups_train.select_dtypes(include=['number']).columns)

# построение корр. тепловой карты
vif_heatmap(
    df=startups_train[num_cols],
    t='Тепловая карта VIF'
)


# Удаление признака с более высоким VIF:

# In[91]:


startups_train = startups_train.drop(columns=['last_funding_at'])

# проверка
startups_train.columns


# **Вывод:**
# 
# С помощью коэффициента VIF была выявлена мультиколлинеарность между `first_funding_at` (1208.433) и `last_funding_at` (1325.853), так как это может сказаться на итоговом качестве модели, то было решено удалить признак с более высоким VIF (`last_funding_at`).

# ## Обучение моделей

# ### Создание выборок

# Целевой признак:

# In[92]:


# определение целевого признака
target_col = 'status'

# вывод целевого признака
print(f'Целевой признак:\n{target_col}\n')


# Входные признаки:

# In[93]:


# числовые входные признаки
num_cols = list(startups_train.select_dtypes(include=[np.number]))

# вывод входных признаков
print(f'Числовые входные признаки:\n{num_cols}\n')


# Разделение данных на выборки и кодирование целевого признака:

# In[94]:


# входные признаки
X_train = startups_train[num_cols]
X_test = startups_test[num_cols]

# кодирование целевого признака
label_encoder = LabelEncoder()
label_encoder.fit(['closed', 'operating'])
y_train = label_encoder.transform(startups_train[target_col])

# проверка кодирования целевого признака
print('Целевой признак закодирован следующим образом:')
print(f'0 - {list(label_encoder.classes_)[0]}')
print(f'1 - {list(label_encoder.classes_)[1]}')


# Устранение дисбаланса категорий целевого признака c помощью метода `SMOTE`:

# In[95]:


# экземпляр сэмплера
sampler = SMOTE(random_state=RANDOM_STATE, k_neighbors=5)

# сэмплирование данных
X_train, y_train = sampler.fit_resample(X_train, y_train)

# проверка
pie_plot(df=pd.DataFrame(y_train, columns=['status']),
         col='status',
         lt='Статусы',
         t='Распределение статусов стартапов в тренировочной выборке после устранения дисбаланса'
)


# Размеры выборок:

# In[96]:


print(f'Размеры входных признаков: тренировочная - {X_train.shape}, тестовая - {X_test.shape}')
print(f'Размеры целевого признака: тренировочная - {y_train.shape}')


# **Вывод:**
# 
# В качестве целевого признака был выбран статус стартапов (`status`), а в качестве входных: общая сумма финансирования (`funding_total_usd`), кол-во раундов финансирования (`funding_rounds`), дата первого раунда финансирования (`first_funding_at`), широта (`latitude`), долгота (`longitude`), PCA-признаки (`category_pca_0`, `category_pca_1`, `category_pca_2`, `category_pca_3`, `category_pca_4`, `category_pca_5`, `category_pca_6`, `category_pca_7`, `category_pca_8`, `category_pca_9`).
# 
# Для дальнейшего обучения моделей было произведено кодирование (label encoder) целевого признака: закрывшийся стартап обозначается как `0` (`closed`), а действующий — как `1` (`operating`).
# 
# Так как в тренировочной выборке наблюдался огромный дисбаланс категорий целевого признака, то было решено его устранить с помощью метода `SMOTE` (кол-во соседей — 5).
# 
# В итоге, выборки имеют следующие размеры:
# 
# - тренировочная выборка: входные признаки имеют 95 056 строк и 15 столбцов (`X_train`), а целевой — 95 056 строк (`y_train`);
# - тестовая выборка: входные признаки имеют 13 125 строк и 15 столбцов (`X_test`).

# ### Пайплайны и гипермараметры

# #### Простые модели

# Пайплайны моделей:

# In[97]:


# логистическая регрессия
pipe_lr = Pipeline([
    ('preprocessor', StandardScaler()),
    ('model', LogisticRegression(solver='liblinear',
                                 random_state=RANDOM_STATE))
])

# логистическая регрессия с L2-регуляризацией
pipe_lr_l2 = Pipeline([
    ('preprocessor', StandardScaler()),
    ('model', LogisticRegression(solver='liblinear', penalty='l2',
                                 random_state=RANDOM_STATE))
])

# логистическая регрессия с L1-регуляризацией
pipe_lr_l1 = Pipeline([
    ('preprocessor', StandardScaler()),
    ('model', LogisticRegression(solver='liblinear', penalty='l1',
                                 random_state=RANDOM_STATE))
])

# дерево решений
pipe_tree = Pipeline([
    ('preprocessor', 'passthrough'),
    ('model', DecisionTreeClassifier(random_state=RANDOM_STATE))
])


# Задание гиперпараметров:

# In[98]:


# гиперпараметры для логистической регрессии
param_distributions_lr = {
    # 'preprocessor': CategoricalDistribution(
        # [
            # 'passthrough',
            # StandardScaler(),
            # MinMaxScaler(),
            # RobustScaler()
        # ]
    # )
}

# гиперпараметры для логистической регрессии с L2-регуляризацией
param_distributions_lr_l2 = {
    # 'preprocessor': CategoricalDistribution(
        # [
            # 'passthrough',
            # StandardScaler(),
            # MinMaxScaler(),
            # RobustScaler()
        # ]
    # ),
    'model__C': FloatDistribution(1e-2, 1e2, log=True)
}

# гиперпараметры для логистической регрессии с L1-регуляризацией
param_distributions_lr_l1 = {
    # 'preprocessor': CategoricalDistribution(
        # [
            # 'passthrough',
            # StandardScaler(),
            # MinMaxScaler(),
            # RobustScaler()
        # ]
    # ),
    'model__C': FloatDistribution(1e-2, 1e2, log=True)
}

# гиперпараметры для дерева решений
param_distributions_tree = {
    'preprocessor': CategoricalDistribution(['passthrough']),
    'model__max_depth': IntDistribution(2, 20),
    'model__min_samples_split': IntDistribution(2, 20),
    'model__min_samples_leaf': IntDistribution(1, 10)
}


# **Вывод:**
# 
# Для подготовки данных простым моделям были выбраны следующие методы масштабирования: `StandardScaler()`, `MinMaxScaler()` и `RobustScaler()`.
# 
# Для задачи классификации было выбрано 4 простых модели машинного обучения:
# 
# - `LogisticRegression()` — логистическая регрессия с гиперпараметрами: обратный коэффициент регуляризации от 0.01 до 100 (больше `C` → слабее регуляризация), алгоритмы оптимизации `liblinear` и `lbfgs` (`solver`);
# - `RidgeClassifier()` — ридж-классификатор с коэффициентом регуляризации от 0.01 до 100 (больше `alpha` → сильнее регуляризация), причем распределение является логарифмическим для лучшего исследования широкого диапазона значений;
# - `LogisticRegression()` — логистическая регрессия с L1-регуляризацией (аналог Lasso) и гиперпараметрами: обратный коэффициент регуляризации от 0.01 до 100 (больше `C` → слабее регуляризация), алгоритмы оптимизации `liblinear` (`lbfgs` не поддерживает L1-регуляризацию);
# - `DecisionTreeClassifier()` — дерево решений с гиперпараметрами: максимальная глубина дерева от 2 до 20 (`max_depth`), минимальное кол-во объектов в узле от 2 до 20 (`min_samples_split`), минимальное кол-во объектов в листе от 1 до 10 (`min_samples_leaf`), мера качества разбиений `gini` и `entropy` (`criterion`).
# 
# Модели вспомогательных векторов, k-ближайших соседей и случайного леса не были взяты, так как они слишком долго обучаются на предоставленных данных.

# #### Градиентный бустинг `LightGBM`

# Пайплайн модели:

# In[99]:


# pipe_lgb = Pipeline([
    # ('model', LGBMRegressor())
# ])


# Задание гиперпараметров:

# In[100]:


# param_distributions_lgb = {
    # 'model__num_leaves': optuna.distributions.IntDistribution(15, 20),
    # 'model__learning_rate': optuna.distributions.FloatDistribution(5e-1, 1, log=True),
    # 'model__max_depth': optuna.distributions.IntDistribution(2, 5)
# }


# **Вывод:**
# 
# Для градиентного бустинга `LightGBM` будут использоваться следующие гиперпараметры:
# 
# - число листьев в дереве: от 15 до 20 (`num_leaves`);
# - шаг градиентного спуска: от 0.5 до 1 (`learning_rate`);
# - максимальная глубина дерева: от 2 до 5 (`max_depth`).

# #### Градиентный бустинг `CatBoost`

# Пайплайн модели:

# In[101]:


# pipe_cb = Pipeline([
    # (
        # 'model', CatBoostRegressor(
            # verbose=0, # чтобы подавить вывод
            # allow_writing_files=False, # отключение ненужного создания файлов
            # cat_features=cat_cols # передача категориальных признаков
        # )
    # )
# ])


# Задание гиперпараметров:

# In[102]:


# param_distributions_cb = {
    # 'model__max_leaves': optuna.distributions.IntDistribution(15, 20),
    # 'model__learning_rate': optuna.distributions.FloatDistribution(5e-1, 1, log=True),
    # 'model__depth': optuna.distributions.IntDistribution(2, 5),
    # 'model__grow_policy': optuna.distributions.CategoricalDistribution(['Lossguide'])
# }


# **Вывод:**
# 
# Для градиентного бустинга `CatBoost` будут использоваться следующие гиперпараметры:
# 
# - число листьев в дереве: от 15 до 20 (`max_leaves`);
# - шаг градиентного спуска: от 0.5 до 1 (`learning_rate`);
# - максимальная глубина дерева: от 2 до 5 (`max_depth`);
# - алгоритм построения деревьев: Lossguide (`grow_policy`).

# ### Кросс-валидация

# #### Простые модели

# Кросс-валидация с помощью `OptunaSearchCV()`:

# In[ ]:


# список моделей, их параметров и описаний
models = [
    {'name': 'Логистическая регрессия', 'pipeline': pipe_lr, 'params': param_distributions_lr},
    {'name': 'Логистическая регрессия с L2-регуляризацией', 'pipeline': pipe_lr_l2, 'params': param_distributions_lr_l2},
    {'name': 'Логистическая регрессия с L1-регуляризацией', 'pipeline': pipe_lr_l1, 'params': param_distributions_lr_l1},
    {'name': 'Дерево решений', 'pipeline': pipe_tree, 'params': param_distributions_tree}
]

# цикл для перебора моделей
optuna_results = {}
for model in models:
    model_name = model['name']
    pipeline = model['pipeline']
    params = model['params']
    
    # вызов функции optuna_val для каждой модели
    optuna_results[model_name] = optuna_val(
        pipeline,
        params,
        scoring='f1',
        X_tr=X_train,
        y_tr=y_train,
        text=f'{model_name}:'
    )


# In[ ]:





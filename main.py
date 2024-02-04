import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats


def lab_1():
    print("-----------lab_1---------")
    url = 'uber.csv'
    data = pd.read_csv(url, na_values='')

    data_sample = data.sample(n=1000, random_state=22)

    missing_values = data_sample.isnull().sum()
    print("Пропущенные значения в датасете:")
    print(missing_values)

    numerical_columns = data_sample.select_dtypes(include=[np.number]).columns.tolist()
    for col in numerical_columns:
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        sns.boxplot(x=col, data=data_sample, orient='v')
        plt.yscale('log')
        plt.title(f'Ящик с усами для {col} (логарифмическая шкала)')
        plt.subplot(1, 2, 2)
        sns.histplot(data_sample[col], bins=20, kde=True,
                     orientation='vertical')  # Добавляем параметр orientation='vertical'
        plt.title(f'Гистограмма для {col}')
        plt.tight_layout()
        plt.show()

    numeric_columns_excluding_datetime = data_sample.select_dtypes(include=[np.number]).columns.tolist()
    data_sample.fillna(data_sample[numeric_columns_excluding_datetime].median(), inplace=True)

    pivot_table = pd.pivot_table(data_sample, index='passenger_count', values='fare_amount', aggfunc="mean")
    print("Сводная таблица:")
    print(pivot_table)

    def task_2():
        def l(x, c):
            return np.power(np.abs(2 * x - c) ** 3, 1 / 5) + 0.567

        x = 12.1

        c_values = np.arange(-10, 1.5, 0.5)

        l_values = l(x, c_values)

        max_value = np.max(l_values)
        min_value = np.min(l_values)
        mean_value = np.mean(l_values)
        array_length = len(l_values)

        print("Максимальное значение функции: ", max_value)
        print("Минимальное значение функции: ", min_value)
        print("Среднее значение функции: ", mean_value)
        print("Количество элементов массива: ", array_length)

        sorted_l_values = np.sort(l_values)

        plt.figure(figsize=(10, 6))
        plt.plot(c_values, l_values, marker='o', label='Значения функции l')

        plt.axhline(y=mean_value, color='r', linestyle='--', label='Среднее значение')

        plt.xlabel('Значение параметра c')
        plt.ylabel('Значение функции l')
        plt.title('График изменения значений функции l')
        plt.legend()
        plt.grid(True)

        plt.show()

    print("--------task 2------")
    task_2()


def lab_2():
    print("--------lab 2------")
    electric_activity = [0, 38.5, 59, 97.4, 119.2, 129.5, 198.7, 248.7, 318, 438.5]
    vascular_permeability = [19.5, 15, 13.5, 23.3, 6.3, 2.5, 13, 1.8, 6.5, 1.8]

    data = pd.DataFrame({'Electric Activity': electric_activity,
                         'Vascular Permeability': vascular_permeability})  # 1 - парам это наклон, а второй интерспета

    correlation_coefficient = np.corrcoef(data['Electric Activity'], data['Vascular Permeability'])[0, 1]

    slope, intercept, r_value, p_value, std_err = stats.linregress(data['Electric Activity'],
                                                                   data['Vascular Permeability'])
    plt.figure(figsize=(10, 6))
    plt.scatter(data['Electric Activity'], data['Vascular Permeability'], color='blue', label='Data points')
    plt.plot(data['Electric Activity'], slope * data['Electric Activity'] + intercept, color='red',
             label='Regression line')
    plt.xlabel('Electric Activity')
    plt.ylabel('Vascular Permeability')
    plt.title('График зависимости')
    plt.legend()
    plt.grid(True)

    if abs(correlation_coefficient) >= 0.7:
        correlation_strength = "высокая"
    elif abs(correlation_coefficient) >= 0.5:
        correlation_strength = "заметная"
    elif abs(correlation_coefficient) >= 0.3:
        correlation_strength = "умеренная"
    elif abs(correlation_coefficient) >= 0.1:
        correlation_strength = "слабая"
    elif abs(correlation_coefficient) >= 0.9:
        correlation_strength = "весьма высокая"
    else:
        correlation_strength = "очень слабая или отсутствует"


    print("Коэффициент корреляции:", correlation_coefficient)
    print("Сила связи:", correlation_strength)

    if p_value < 0.05:
        correlation_significance = "значимая"
    else:
        correlation_significance = "незначимая"

    print("Значимость корреляции:", correlation_significance)
    print("Уравнение регрессии: Vascular Permeability = {:.2f} * Electric Activity + {:.2f}".format(slope, intercept))

    plt.show()

    print("---------------tast 2-------------------")
    data = {
        'n': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        'X': [26, 48, 65, 23, 150, 123, 264, 156, 152, 154, 520, 415],
        'y': [221, 153, 155, 102, 156, 264, 435, 156, 203, 325, 456, 163]
    }

    df = pd.DataFrame(data)

    mean_X = np.mean(df['X'])
    mean_y = np.mean(df['y'])

    df['XY_product'] = df['X'] * df['y']
    df['X_squared'] = df['X'] ** 2
    df['Y_squared'] = df['y'] ** 2
    df['X_minus_mean'] = df['X'] - mean_X
    df['X_minus_mean_squared'] = df['X_minus_mean'] ** 2

    covariance = np.sum((df['X'] - mean_X) * (df['y'] - mean_y))
    variance = np.sum((df['X'] - mean_X) ** 2)

    b1 = covariance / variance
    b0 = mean_y - b1 * mean_X

    print(f"Уравнение регрессии: y = {b0:.2f} + {b1:.2f} * X")

    plt.scatter(df['X'], df['y'], color='blue', label='Данные')
    plt.plot(df['X'], b0 + b1 * df['X'], color='red', label='Линия тренда')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('Корреляционное поле и линия тренда')
    plt.legend()
    plt.show()

    print("Таблица с промежуточными значениями:")
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    print(df)

    df.to_csv('intermediate_table.csv', index=False)



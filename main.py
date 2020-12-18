import numpy
import pandas
import matplotlib

mean_x1 = 0
range = 0


def linear_regression(dataset):
    c1 = 0
    c2 = 0
    return gradient_descent(c1, c2, dataset)


def gradient_descent(c1, c2, g_dataset):
    learning_rate = 0.001
    old_error = mean_sqr_error(g_dataset['price'], calculate_y_pre(c1, c2, g_dataset['sqft_living15']))
    while True:
        temp_c1 = c1 - diff_cost_function_c1(c1, c2, g_dataset) * learning_rate
        temp_c2 = c2 - diff_cost_function_c2(c1, c2, g_dataset) * learning_rate
        c1 = temp_c1
        c2 = temp_c2
        new_error = mean_sqr_error(g_dataset['price'], calculate_y_pre(c1, c2, g_dataset['sqft_living15']))
        if old_error == new_error:
            break
        old_error = new_error
    return [c1, c2]


def hypothesis(c1, c2, x1):
    return c1 + c2 * scale_x1(x1)


def calculate_y_pre(c1, c2, x1):
    y_pre = list()
    for i in x1:
        y_pre.append(hypothesis(c1, c2, scale_x1(i)))
    return y_pre


def diff_cost_function_c1(c1, c2, dataset):
    sum = 0.0
    for i in range(0, len(dataset)):
        y = dataset['price'][i]
        y_predict = hypothesis(c1, c2, scale_x1(dataset['sqft_living15'][i]))
        sum = sum + (y_predict - y)
    return sum / len(dataset)


def diff_cost_function_c2(c1, c2, dataset):
    sum = 0.0
    for i in range(0, len(dataset)):
        y = dataset['price'][i]
        y_predict = hypothesis(c1, c2, scale_x1(dataset['sqft_living15'][i]))
        sum = sum + ((y_predict - y) * scale_x1(dataset['sqft_living15'][i]))
    return sum / len(dataset)


def mean_sqr_error(y, y_pre):
    sum = 0.0
    length = len(y_pre)
    for i in range(0, length):
        sum = sum + numpy.power((y_pre[i] - y[i]), 2)
    return sum / (2 * len(y))


def scale_x1(x1):
    return (x1 - mean_x1) / range


if __name__ == '__main__':
    intiale_dataset = pandas.read_csv("house_data.csv")
    mean_x1 = intiale_dataset['sqft_living15'].mean()
    range = intiale_dataset['sqft_living15'].max() - intiale_dataset['sqft_living15'].min()
    size_of_dataset = len(intiale_dataset)
    size_of_testet = int(size_of_dataset * 0.8)
    print(size_of_testet)
    parameters = linear_regression(intiale_dataset[:size_of_testet])
    outputs = list()
    sum = 0.0
    for i in range(size_of_testet, size_of_dataset):
        y_pre = hypothesis(parameters[0], parameters[1], scale_x1(intiale_dataset['sqft_living15'][i]))
        y = intiale_dataset['price'][i]
        sum = sum + numpy.power(y_pre - y, 2)
    print(sum)

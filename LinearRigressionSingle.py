import numpy
import pandas
import matplotlib.pyplot as plt


def main():
    dataset = pandas.read_csv("house_data.csv")
    needed_dataset_size = int(len(dataset) * 0.8)
    training_set = dataset[0:needed_dataset_size]
    mean_of_dataset_x1 = dataset['sqft_living'].mean()
    range_of_dataset_x1 = dataset['sqft_living'].max() - dataset['sqft_living'].min()
    mean_of_dataset_price = dataset['price'].mean()
    range_of_dataset_price = dataset['price'].max() - dataset['price'].min()
    x1_training_set = (training_set['sqft_living'] - mean_of_dataset_x1) / (
        range_of_dataset_x1)
    prices_training_set = (training_set['price'] - mean_of_dataset_price) / (
        range_of_dataset_price)
    test_set = dataset[needed_dataset_size:]
    test_set.reset_index(drop=True, inplace=True)
    x1_test_set = (test_set['sqft_living'] - mean_of_dataset_x1) / (
        range_of_dataset_x1)
    prices_test_set = (test_set['price'] - mean_of_dataset_price) / (
        range_of_dataset_price)
    parameters = start_liner_regression(prices_training_set, x1_training_set)

    print("testing error")
    test(prices_test_set, x1_test_set, parameters)

    plt.xlabel("sqft_living")
    plt.ylabel("price")
    plt.title("Real vs predicted values")
    plt.scatter(x1_test_set, prices_test_set)
    plt.plot(x1_test_set, calculate_y_pre(parameters[0], parameters[1], x1_test_set))
    plt.show()
    print("Enter x to exit")
    while True:
        x1 = input("Enter sqft_living")
        if x1.__eq__('x'):
            break
        print(un_scale(hypothesis(parameters[0], parameters[1],
                                  scale_feature(float(x1), mean_of_dataset_x1, range_of_dataset_x1)),
                       mean_of_dataset_price, range_of_dataset_price))


def un_scale(y, mean, my_range):
    return (y * my_range) + mean


def scale_feature(x1, mean, my_range):
    return (x1 - mean) / my_range


def start_liner_regression(prices, x1):
    c1 = 0
    c2 = 1
    return gradient_descent(c1, c2, prices, x1)


def gradient_descent(c1, c2, prices, x1):
    learning_rate = 0.05
    y_pre = calculate_y_pre(c1, c2, x1)
    old_cost = cost_function(prices, y_pre)
    number_of_iterations = 2000
    counter = 0
    while True:
        temp_c1 = c1 - learning_rate * diff_cost_function_c1(prices, y_pre)
        temp_c2 = c2 - learning_rate * diff_cost_function_c2(prices, y_pre, x1)
        c1 = temp_c1
        c2 = temp_c2
        y_pre = calculate_y_pre(c1, c2, x1)
        new_cost = cost_function(prices, y_pre)
        counter = counter + 1
        # print(old_cost)
        if counter == number_of_iterations:
            break
        old_cost = new_cost
    print("c1: ", c1)
    print("c2: ", c2)
    return [c1, c2]


def cost_function(y, y_pre):
    sum = 0.0
    for i in range(0, len(y)):
        sum = sum + numpy.power((y_pre[i] - y[i]), 2)
    return sum / (2 * len(y))


def hypothesis(c1, c2, x1):
    return c1 + c2 * x1


def calculate_y_pre(c1, c2, x1):
    y_pre = list()
    for i in x1:
        y_pre.append(hypothesis(c1, c2, i))
    return y_pre


def diff_cost_function_c1(y, y_pre):
    sum = 0.0
    for i in range(0, len(y)):
        sum = sum + (y_pre[i] - y[i])
    return sum / len(y)


def diff_cost_function_c2(y, y_pre, x1):
    sum = 0.0
    for i in range(0, len(y)):
        sum = sum + ((y_pre[i] - y[i]) * x1[i])
    return sum / len(y)


def test(prices, x1, parameters):
    y_pre = calculate_y_pre(parameters[0], parameters[1], x1)
    print(cost_function(prices, y_pre))


main()

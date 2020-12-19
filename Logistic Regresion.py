import numpy
import pandas
import matplotlib.pyplot as plt

column_name = ['target', 'trestbps', 'chol', 'thalach', 'oldpeak']


def main():
    dataset = pandas.read_csv("heart.csv")
    needed_dataset_size = int(len(dataset) * 0.8)
    training_set = dataset[0:needed_dataset_size]
    test_set = dataset[needed_dataset_size:]
    test_set.reset_index(drop=True, inplace=True)
    mean_list = list()
    range_list = list()
    scaled_training_set_features = list()
    scaled_test_set_features = list()
    for i in range(0, len(column_name)):
        mean_list.append(dataset[column_name[i]].mean())
        range_list.append(dataset[column_name[i]].max() - dataset[column_name[i]].min())
    for i in range(0, len(column_name)):
        scaled_training_set_features.append((training_set[column_name[i]] - mean_list[i]) / range_list[i])
        scaled_test_set_features.append((test_set[column_name[i]] - mean_list[i]) / range_list[i])

    parameters = start_liner_regression(scaled_training_set_features)

    print("testing error")
    test(scaled_test_set_features, parameters)


def un_scale(y, mean, my_range):
    return (y * my_range) + mean


def scale_feature(x1, mean, my_range):
    return (x1 - mean) / my_range


def start_liner_regression(scaled_features):
    parameters = list()
    for i in range(0, len(scaled_features)):
        parameters.append(i)
    return gradient_descent(parameters, scaled_features)


def gradient_descent(parameters, scaled_features):
    learning_rate = 0.001
    y_pre = calculate_y_pre(parameters, scaled_features[1:])
    old_cost = cost_function(scaled_features[0], y_pre)
    number_of_iterations = 10778
    counter = 0
    temp_parameters = [0] * len(parameters)
    while True:
        temp_parameters[0] = parameters[0] - learning_rate * diff_cost_function_c1(scaled_features[0], y_pre)
        for i in range(1, len(temp_parameters)):
            temp_parameters[i] = parameters[i] - learning_rate * diff_cost_function_c2(scaled_features[0], y_pre,
                                                                                       scaled_features[i])

        for i in range(0, len(temp_parameters)):
            parameters[i] = temp_parameters[i]
        y_pre = calculate_y_pre(parameters, scaled_features[1:])
        new_cost = cost_function(scaled_features[0], y_pre)
        counter = counter + 1
        # print(old_cost)
        if old_cost < new_cost:
            print("bom shaka laka")
            break
        if counter == number_of_iterations:
            break
        old_cost = new_cost
    for i in range(0, len(parameters)):
        print("c", i, ": ", parameters[i])
    return parameters


def cost_function(y, y_pre):
    sum = 0.0
    for i in range(0, len(y)):
        sum = sum + numpy.power((y_pre[i] - y.iloc[i]), 2)
    return sum / (2 * len(y))


def linear_equation(parameters, scaled_features):
    sum = parameters[0]
    for i in range(0, len(scaled_features)):
        sum = sum + (parameters[i + 1] * scaled_features[i])
    return sum


def sigmoid(sum):
    return 1 / (1 + numpy.exp(-sum))


def hypothesis(parameters, scaled_features):
    sum = linear_equation(parameters, scaled_features)
    sigmoid_value = sigmoid(sum)
    if sigmoid_value < 0.5:
        return 0
    else:
        return 1


def calculate_y_pre(parameters, scaled_features):
    y_pre = list()
    for i in range(0, len(scaled_features[0])):
        features = list()
        for j in scaled_features:
            features.append(j.iloc[i])
        y_pre.append(hypothesis(parameters, features))
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


def test(scaled_features, parameters):
    y_pre = calculate_y_pre(parameters, scaled_features[1:])
    print(cost_function(scaled_features[0], y_pre))


main()

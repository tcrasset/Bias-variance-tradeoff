import numpy as np
import math
from matplotlib import pyplot as plt
from sklearn.utils import check_random_state
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

def output(x):
    return math.sin(x) + 0.5 * math.sin(3*x)
            
def create_LS(min_bound, max_bound, nb_points, mean, var, nb_ls):
    learning_samples = []
    for i in range(nb_ls):
        x = np.random.uniform(min_bound, max_bound, nb_points)
        x.sort()
        y = [output(x_) + np.random.normal(mean, math.sqrt(var)) for x_ in x] 
        x_as_array = [[x_] for x_ in x]
        learning_samples.append((x_as_array,y))
    return learning_samples

def fit_models(learning_samples, model_type, n_neighbors=None):
    if model_type == "KNR":
        knr_models = []
        for learning_sample in learning_samples:
            (x,y) = learning_sample
            knr = KNeighborsRegressor(n_neighbors)
            knr.fit(x,y)
            knr_models.append(knr)
        return knr_models
    elif model_type == "LNR":
        lnr_models = []
        for learning_sample in learning_samples:
            (x,y) = learning_sample
            lnr = LinearRegression()
            lnr.fit(x,y)
            lnr_models.append(lnr)
        return lnr_models

def get_average_LS(models, x):
    predicted = [model.predict([[x]]) for model in models]
    return np.mean(predicted)

def get_variance_LS(models, x):
    average_lS = get_average_LS(models,x)
    computeVar = lambda pred : (pred - average_lS)**2
    predicted = [model.predict([[x]]) for model in models]
    variances = list(map(computeVar, predicted))
    return np.mean(variances)

def get_squared_bias(models, x):
    return (get_average_LS(models,x) - output(x))**2

def get_residual_error(mean, var, nb_ls, x):
    outputs = [output(x)+ np.random.normal(mean, math.sqrt(var)) for i in range(nb_ls)]
    return np.var(outputs)

def get_expected_error(models, x, residual_error):
    return residual_error  \
            + get_variance_LS(models,x) \
            + get_squared_bias(models,x)

def compute_quantities(x, mean, var, nb_ls, nb_points, model_type, n_neighbors=None):
    learning_samples = create_LS(-4, 4, nb_points, mean, var, nb_ls)
    models = fit_models(learning_samples, model_type, n_neighbors)

    residual_errors = []
    squared_bias = []
    variances = []
    expected_errors = []

    for x_zero in x:
        # print(x_zero)
        residual_error = get_residual_error(mean, var, nb_ls, x_zero)
        residual_errors.append(residual_error)
        squared_bias.append(get_squared_bias(models, x_zero))
        variances.append(get_variance_LS(models, x_zero))
        expected_errors.append(get_expected_error(models, x_zero, residual_error))
    return residual_errors, squared_bias, variances, expected_errors

def mean_quantities(x, mean, var, nb_ls, nb_points, model_type, n_neighbors=None):
   quantities = compute_quantities(x, mean, var, nb_ls, nb_points, "LNR")
   return [np.mean(quantity) for quantity in quantities]

def change_LS_size(x, mean, var, nb_ls, model_type, n_neighbors=None):
    size_LS = range(10,100)
    mean_residual_list = []
    mean_squared_bias_list = []
    mean_variances_list= []
    mean_expected_list = []

    for nb_points in size_LS:
        mean_residual_errors, \
        mean_squared_bias, \
        mean_variances, \
        mean_expected_errors = mean_quantities(x, mean, var, nb_ls, nb_points, model_type, n_neighbors=n_neighbors)

        mean_residual_list.append(mean_residual_errors)
        mean_squared_bias_list.append(mean_squared_bias)
        mean_variances_list.append(mean_variances)
        mean_expected_list.append(mean_expected_errors)
        print(nb_points)
    return size_LS, \
            mean_residual_list, \
            mean_squared_bias_list, \
            mean_variances_list, \
            mean_expected_list

def change_complexity(x, mean, var, nb_ls, nb_points, model_type):
    complexity = range(1, 125,1)
    mean_residual_list = []
    mean_squared_bias_list = []
    mean_variances_list= []
    mean_expected_list = []

    for n_neighbors in complexity:
        mean_residual_errors, \
        mean_squared_bias, \
        mean_variances, \
        mean_expected_errors = mean_quantities(x, mean, var, nb_ls, nb_points, model_type, n_neighbors)

        mean_residual_list.append(mean_residual_errors)
        mean_squared_bias_list.append(mean_squared_bias)
        mean_variances_list.append(mean_variances)
        mean_expected_list.append(mean_expected_errors)
        print(n_neighbors)
    return complexity, \
            mean_residual_list, \
            mean_squared_bias_list, \
            mean_variances_list, \
            mean_expected_list

def change_var(x, mean, nb_ls, nb_points, model_type, n_neighbors=None):
    variance = range(0.001,1,0.01)
    mean_residual_list = []
    mean_squared_bias_list = []
    mean_variances_list= []
    mean_expected_list = []

    for var in variance:
        mean_residual_errors, \
        mean_squared_bias, \
        mean_variances, \
        mean_expected_errors = mean_quantities(x, mean, var, nb_ls, nb_points, model_type, n_neighbors=n_neighbors)

        mean_residual_list.append(mean_residual_errors)
        mean_squared_bias_list.append(mean_squared_bias)
        mean_variances_list.append(mean_variances)
        mean_expected_list.append(mean_expected_errors)
        print(nb_points)
    return variance, \
            mean_residual_list, \
            mean_squared_bias_list, \
            mean_variances_list, \
            mean_expected_list

def draw_plot(title,x , xlabel,y , ylabel, filename = None):
    plt.figure()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.plot(x,y, label=ylabel)

    plt.legend(loc='upper right')

    #plt.savefig(filename)
    plt.show()
    plt.close()

if __name__ == "__main__":
    mean = 0
    var = 0.01
    nb_ls = 1
    nb_points = 1000
    x = np.arange(-4, 4, 0.01)


    # residual_errors, \
    # squared_bias, \
    # variances, \
    # expected_errors = compute_quantities(x, mean, var, nb_ls, nb_points, "LNR")

    # draw_plot("Residual error", x, "x", residual_errors, "residual_errors")
    # draw_plot("Squared bias", x, "x", squared_bias, "squared_bias")
    # draw_plot("Variances", x, "x", variances, "variances")
    # draw_plot("Expected error", x, "x", expected_errors, "expected_errors")
    
    # residual_errors, \
    # squared_bias, \
    # variances, \
    # expected_errors = compute_quantities(x, mean, var, nb_ls, nb_points, "KNR",5)
    # draw_plot("Residual error", x, "x", residual_errors, "residual_errors")
    # draw_plot("Squared bias", x, "x", squared_bias, "squared_bias")
    # draw_plot("Variances", x, "x", variances, "variances")
    # draw_plot("Expected error", x, "x", expected_errors, "expected_errors")
 
 
    size_LS, \
    residual_errors, \
    squared_bias, \
    variances, \
    expected_errors = change_LS_size(x, mean, var, 3, nb_points, "LNR")
    draw_plot("Residual error", size_LS, "size_LS", residual_errors, "residual_errors")
    draw_plot("Squared bias", size_LS, "size_LS", squared_bias, "squared_bias")
    draw_plot("Variances", size_LS, "size_LS", variances, "variances")
    draw_plot("Expected error", size_LS, "size_LS", expected_errors, "expected_errors")

    # complexity, \
    # residual_errors, \
    # squared_bias, \
    # variances, \
    # expected_errors = change_complexity(x, mean, var, 3, nb_points, "KNR")
    # draw_plot("Residual error", complexity, "complexity", residual_errors, "residual_errors")
    # draw_plot("Squared bias", complexity, "complexity", squared_bias, "squared_bias")
    # draw_plot("Variances", complexity, "complexity", variances, "variances")
    # draw_plot("Expected error", complexity, "complexity", expected_errors, "expected_errors")



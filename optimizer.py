import optuna
from main import TimeOutException, main

def objective(trial: optuna.Trial):

    learning_rate = trial.suggest_float('Learning Rate', 0.0001, 0.01) #suggest log uniform
    #learning_rate = 0.00026
    nn_layers_size_exp = trial.suggest_int('NN Layer Size Exponent', 4, 10)
    #nn_layers_size_exp = 7
    nn_layers_size = 2 ** nn_layers_size_exp

    result = main(learning_rate=learning_rate, nn = nn_layers_size, save_run=False)

    return result

study = optuna.create_study()
TWO_MINUTES = 60 * 2
study.optimize(objective, n_trials=20, catch=(TimeOutException,))

print(study.best_params)
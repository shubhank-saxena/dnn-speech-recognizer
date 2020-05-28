# allocate 50% of GPU memory (if you like, feel free to change this)
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf 
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))

# import NN architectures for speech recognition
from sample_models import *
# import function for training acoustic model
from train_utils import train_model_history

import GPy
import GPyOpt

from keras.optimizers import SGD
import numpy as np

def best_validation_loss(hist):
    print("hist: {}".format(hist))
    val_loss = min([1e6 if np.isnan(x) else x for x in hist['val_loss']])
    print('val_loss: {}'.format(val_loss))
    return val_loss

def print_params(model_name,x,names):
    print("params for {}:".format(model_name))
    for n,v in zip(names,x):
        print("    {}: {}".format(n,v))

def print_objective(model_name,obj):
    print("objective for {}: {}".format(model_name, obj))
        
def opt_params_for_rnn_model():
    bounds = [
        {'name': 'units', 'type': 'continuous', 'domain': (50,400)},
        {'name': 'lr', 'type': 'continuous', 'domain': (0,0.1)},
        {'name': 'dropout_rate', 'type': 'continuous', 'domain': (0,0.5)}
    ]
    
    def objective(x):
        units, lr, dropout_rate = x[0]
        units = int(units)
        model = rnn_model(input_dim=161,units=units,activation='relu',dropout_rate=dropout_rate)
        hist = train_model_history(
            input_to_softmax=model, 
            optimizer=SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=1.0),
            epochs=20,
            spectrogram=True)
        value = best_validation_loss(hist.history)
        print_params('rnn', x[0], [b['name'] for b in bounds])
        print_objective('rnn',value)
        return value
    
    max_iter = 10
    problem = GPyOpt.methods.BayesianOptimization(objective,bounds)
    problem.run_optimization(max_iter)
    print('Optimized rnn:')
    print_params('rnn',problem.x_opt, [b['name'] for b in bounds])
    print_objective('rnn',problem.fx_opt)
    
def opt_params_for_cnn_rnn_model():
    bounds = [
        {'name': 'filters', 'type': 'continuous', 'domain': (10,200)},
        {'name': 'units', 'type': 'continuous', 'domain': (50,400)},
        {'name': 'lr', 'type': 'continuous', 'domain': (0,0.1)},
        {'name': 'dropout_rate', 'type': 'continuous', 'domain': (0,0.5)}
    ]
    
    def objective(x):
        filters, units, lr, dropout_rate = x[0]
        filters = int(filters)
        units = int(units)
        model = cnn_rnn_model(
            input_dim=161,
            filters=filters,
            kernel_size=5, 
            conv_stride=3,
            conv_border_mode='valid',
            units=units,
            dropout_rate=dropout_rate)
        hist = train_model_history(
            input_to_softmax=model, 
            optimizer=SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=1.0),
            epochs=20,
            spectrogram=True)
        value = best_validation_loss(hist.history)
        print_params('rnn', x[0], [b['name'] for b in bounds])
        print_objective('rnn',value)
        return value
    
    max_iter = 10
    problem = GPyOpt.methods.BayesianOptimization(objective,bounds)
    problem.run_optimization(max_iter)
    print('Optimized cnn_rnn:')
    print_params('cnn_rnn',problem.x_opt, [b['name'] for b in bounds])
    print_objective('cnn_rnn',problem.fx_opt)

def opt_params_for_deep_rnn_model():
    bounds = [
        {'name': 'units', 'type': 'continuous', 'domain': (10,300)},
        {'name': 'recur_layers', 'type': 'discrete', 'domain': (1,2,3,4)},
        {'name': 'lr', 'type': 'continuous', 'domain': (0,0.1)},
        {'name': 'dropout_rate', 'type': 'continuous', 'domain': (0,0.5)}
    ]
    
    def objective(x):
        units, recur_layers, lr, dropout_rate = x[0]
        units = int(units)
        recur_layers = int(recur_layers)
        model = deep_rnn_model(input_dim=161,units=units,recur_layers=recur_layers,dropout_rate=dropout_rate)
        hist = train_model_history(
            input_to_softmax=model, 
            optimizer=SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=1.0),
            epochs=20,
            spectrogram=True)
        value = best_validation_loss(hist.history)
        print_params('deep_rnn', x[0], [b['name'] for b in bounds])
        print_objective('deep_rnn',value)
        return value
    
    max_iter = 10
    problem = GPyOpt.methods.BayesianOptimization(objective,bounds)
    problem.run_optimization(max_iter)
    print('Optimized deep_rnn:')
    print_params('deep_rnn',problem.x_opt, [b['name'] for b in bounds])
    print_objective('deep_rnn',problem.fx_opt)

def opt_params_for_bidirectional_rnn_model():
    bounds = [
        {'name': 'units', 'type': 'continuous', 'domain': (10,300)},
        {'name': 'lr', 'type': 'continuous', 'domain': (0,0.1)},
        {'name': 'dropout_rate', 'type': 'continuous', 'domain': (0,0.5)}
    ]
    
    def objective(x):
        units, lr, dropout_rate = x[0]
        units = int(units)
        model = bidirectional_rnn_model(input_dim=161,units=units,dropout_rate=dropout_rate)
        hist = train_model_history(
            input_to_softmax=model, 
            optimizer=SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=1.0),
            epochs=20,
            spectrogram=True)
        value = best_validation_loss(hist.history)
        print_params('bidirectional_rnn', x[0], [b['name'] for b in bounds])
        print_objective('bidirectional_rnn',value)
        return value
    
    max_iter = 10
    problem = GPyOpt.methods.BayesianOptimization(objective,bounds)
    problem.run_optimization(max_iter)
    print('Optimized bidirectional_rnn:')
    print_params('bidirectional_rnn',problem.x_opt, [b['name'] for b in bounds])
    print_objective('bidirectional_rnn',problem.fx_opt)
    
def main():
    #opt_params_for_rnn_model()
    #opt_params_for_cnn_rnn_model()
    opt_params_for_deep_rnn_model()
    opt_params_for_bidirectional_rnn_model
    
if __name__== "__main__":
    main()
    
        

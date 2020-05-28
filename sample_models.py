from keras import backend as K
from keras.models import Model, Sequential
from keras.layers import (BatchNormalization, Conv1D, Conv2D, Dense, Input, Reshape, Lambda,
                          MaxPooling2D,GlobalAveragePooling1D,Dropout,RepeatVector, Flatten,
                          TimeDistributed, Activation, Bidirectional, SimpleRNN, GRU, LSTM)

def simple_rnn_model(input_dim, output_dim=29):
    """ Build a recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layer
    simp_rnn = GRU(
        output_dim,
        return_sequences=True, 
        implementation=2, 
        name='rnn')(input_data)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(simp_rnn)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def rnn_model(input_dim, units, activation, output_dim=29, dropout_rate=0.5):
    """ Build a recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layer
    simp_rnn = GRU(
        units,
        activation=activation,
        return_sequences=True,
        implementation=2,
        name='rnn',
        recurrent_dropout=dropout_rate,
        dropout=dropout_rate)(input_data)
    bn_rnn = BatchNormalization()(simp_rnn)
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model


def cnn_rnn_model(input_dim, filters, kernel_size, conv_stride,
    conv_border_mode, units, output_dim=29, dropout_rate=0.5):
    """ Build a recurrent + convolutional network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add convolutional layer
    conv_1d = Conv1D(filters, kernel_size, 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d')(input_data)
    # Add batch normalization
    bn_cnn = BatchNormalization(name='bn_conv_1d')(conv_1d)
    # Add a recurrent layer
    simp_rnn = GRU(
        units,
        activation='relu',
        return_sequences=True,
        implementation=2,
        name='rnn',
        recurrent_dropout=dropout_rate,
        dropout=dropout_rate)(bn_cnn)
    # TODO: Add batch normalization
    bn_rnn = BatchNormalization()(simp_rnn)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)
    print(model.summary())
    return model

def cnn_output_length(input_length, filter_size, border_mode, stride,
                       dilation=1):
    """ Compute the length of the output sequence after 1D convolution along
        time. Note that this function is in line with the function used in
        Convolution1D class from Keras.
    Params:
        input_length (int): Length of the input sequence.
        filter_size (int): Width of the convolution kernel.
        border_mode (str): Only support `same` or `valid`.
        stride (int): Stride size used in 1D convolution.
        dilation (int)
    """
    if input_length is None:
        return None
    assert border_mode in {'same', 'valid'}
    dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
    if border_mode == 'same':
        output_length = input_length
    elif border_mode == 'valid':
        output_length = input_length - dilated_filter_size + 1
    return (output_length + stride - 1) // stride

def deep_rnn_model(input_dim, units, recur_layers, output_dim=29, dropout_rate=0.5):
    """ Build a deep recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    bn_rnn = input_data
    for _ in range(recur_layers):
        bn_rnn = GRU(
            units, 
            activation='relu',
            return_sequences=True,
            implementation=2,
            recurrent_dropout=dropout_rate,
            dropout=dropout_rate)(bn_rnn)
        bn_rnn = BatchNormalization()(bn_rnn)
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def bidirectional_rnn_model(input_dim, units, output_dim=29, dropout_rate=0.5):
    """ Build a bidirectional recurrent network for speech
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    bidir_rnn = Bidirectional(
        GRU(
            units, 
            activation='relu',
            return_sequences=True,
            implementation=2,
            name='rnn',
            recurrent_dropout=dropout_rate,
            dropout=dropout_rate))(input_data)
    time_dense = TimeDistributed(Dense(output_dim))(bidir_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def cnn2d_rnn_model(input_dim, filters, kernel_size, conv_stride, conv_border_mode, pool_size, units, output_dim=29, dropout_rate=0.5):
    """ Build a bidirectional recurrent network for speech
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    nn = Lambda(lambda y: K.expand_dims(y, -1))(input_data)
    # Add convolutional layer
    nn = Conv2D(filters, kernel_size, padding=conv_border_mode, activation='relu')(nn)
    nn = MaxPooling2D(pool_size)(nn)
    nn = TimeDistributed(Flatten())(nn)
    # Add a recurrent layer
    nn = GRU(
        units,
        activation='relu',
        return_sequences=True,
        implementation=2,
        recurrent_dropout=dropout_rate,
        dropout=dropout_rate)(nn)
    nn = TimeDistributed(Dense(output_dim))(nn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(nn)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    
    f1 = lambda x: cnn_output_length(x, 5, conv_border_mode, conv_stride)
    f2 = lambda x: cnn_output_length(x, conv_stride, conv_border_mode, conv_stride)
    
    model.output_length = lambda x: x
    print(model.summary())
    return model

def final_model(input_dim=161, filters=50, kernel_size=(11,11), conv_stride=1, conv_border_mode='same', pool_size=(1,5), units=200, recur_layers=2, output_dim=29, dropout_rate=0.50):
    """ Build a deep network for speech 
    """    
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    nn = Lambda(lambda y: K.expand_dims(y, -1))(input_data)
    # Add convolutional layer
    nn = Conv2D(filters, kernel_size, padding=conv_border_mode, activation='relu')(nn)
    nn = MaxPooling2D(pool_size)(nn)
    nn = TimeDistributed(Flatten())(nn)
    
    for _ in range(recur_layers):
        nn = Bidirectional(
            GRU(
                units,
                activation='relu',
                return_sequences=True,
                implementation=2,
                recurrent_dropout=dropout_rate,
                dropout=dropout_rate))(nn)
        nn = BatchNormalization()(nn)
    time_dense = TimeDistributed(Dense(output_dim))(nn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model
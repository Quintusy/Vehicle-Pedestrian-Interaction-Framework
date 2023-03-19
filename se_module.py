import random
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import time
import keras

from math import ceil
from prettytable import PrettyTable
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
from os.path import join
from keras import regularizers
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from pandas import read_csv
from keras.models import load_model
from keras.models import Model
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau
from keras.layers.recurrent import LSTM
from keras.layers import Dense, warnings
from keras.layers import Flatten
from keras.layers import BatchNormalization
from keras.layers import Dropout
from keras.layers import Input
from keras.layers import RepeatVector
from keras.layers import Multiply
from keras.layers import Permute
from utils import *


class SE_Module(Model):
    def __init__(self,
                 encoder_feature_size=2,
                 prediction_size=1,
                 observe_length=15,
                 num_hidden_units=64,
                 regularizer_val=0.0001,
                 activation='softsign',
                 embed_size=16,
                 embed_dropout=0.35):
        super(SE_Module, self).__init__()

        self._encoder_feature_size = encoder_feature_size
        self._prediction_size = prediction_size  # 2
        self._observe_length = observe_length

        self._num_hidden_units = num_hidden_units
        self._regularizer_value = regularizer_val
        self._regularizer = regularizers.l2(regularizer_val)

        self._activation = activation
        self._embed_size = embed_size
        self._embed_dropout = embed_dropout

    def create_lstm_model(self, name='lstm', r_state=True, r_sequence=True):

        return LSTM(units=self._num_hidden_units,
                    return_state=r_state,
                    return_sequences=r_sequence,
                    recurrent_dropout=0.3,
                    stateful=False,
                    kernel_regularizer=self._regularizer,
                    recurrent_regularizer=self._regularizer,
                    bias_regularizer=self._regularizer,
                    activity_regularizer=None,
                    activation=self._activation,
                    name=name)

    def attention_temporal(self, input_data, sequence_length):

        a = Permute((2, 1))(input_data)
        a = Dense(sequence_length, activation='sigmoid')(a)
        a_probs = Permute((2, 1))(a)
        output_attention_mul = Multiply()([input_data, a_probs])
        return output_attention_mul

    def attention_element(self, input_data, input_dim):

        input_data_probs = Dense(input_dim, activation='sigmoid')(input_data)
        output_attention_mul = Multiply()([input_data, input_data_probs])

        return output_attention_mul

    def SEmodel(self):

        _encoder_input = Input(shape=(self._observe_length, self._encoder_feature_size),
                               name='encoder_input')

        _attention_net = self.attention_temporal(_encoder_input, self._observe_length)

        encoder_model = self.create_lstm_model(name='encoder_network')
        _encoder_outputs_states = encoder_model(_attention_net)
        _encoder_states = _encoder_outputs_states[1:]

        decoder_model = self.create_lstm_model(name='decoder_network', r_state=False)
        _hidden_input = RepeatVector(self._observe_length)(_encoder_states[0])

        _embedded_hidden_input = Dense(self._embed_size, activation='relu')(_hidden_input)
        _embedded_hidden_input = Dropout(self._embed_dropout,
                                         name='dropout_dec_input')(_embedded_hidden_input)

        att_input_dim = self._embed_size
        decoder_concat_inputs = self.attention_element(_embedded_hidden_input, att_input_dim)

        decoder_output = decoder_model(decoder_concat_inputs,
                                       initial_state=_encoder_states)

        bn_layer = BatchNormalization()(decoder_output)

        flatten_layer = Flatten(name='flatten_layer')(bn_layer)

        pred = Dense(self._prediction_size, activation='sigmoid', name='feature')(flatten_layer)

        net_model = Model(inputs=_encoder_input, outputs=pred)

        net_model.summary()

        return net_model

    def get_model(self):
        train_model = self.SEmodel()
        return train_model

    def get_path(self,
                 type_save='models',
                 models_save_folder='',
                 model_name='convlstm_encdec',
                 file_name='',
                 data_subset='',
                 data_type='',
                 save_root_folder='models/'):
        # save_root_folder=os.environ['PIE_PATH'] + '/models/'):

        assert (type_save in ['models', 'data'])
        if data_type != '':
            assert (any([d in data_type for d in ['images', 'features']]))
        root = os.path.join(save_root_folder, type_save)

        if type_save == 'models':
            save_path = os.path.join(save_root_folder, 'SE_Module', models_save_folder)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            return os.path.join(save_path, file_name), save_path
        else:
            save_path = os.path.join(root, 'SE_Module', data_subset, data_type, model_name)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            return save_path

    def get_model_config(self):

        config = dict()
        config['encoder_feature_siz'] = self._encoder_feature_size
        # config['bias_init'] = self._bias_initializer
        config['prediction_size'] = self._prediction_size
        config['observe_length'] = self._observe_length
        config['num_hidden_units'] = self._num_hidden_units
        config['regularizer_value'] = self._regularizer_value
        config['regularizer'] = self._regularizer
        config['activation'] = self._activation
        config['embed_size'] = self._embed_size
        config['embed_dropout'] = self._embed_dropout

        print(config)
        return config

    def load_model_config(self, config):

        self._encoder_feature_size = config['encoder_feature_siz']
        self._prediction_size = config['prediction_size']
        self._observe_length = config['observe_length']

        self._num_hidden_units = config['num_hidden_units']
        self._regularizer_value = config['regularizer_value']
        self._regularizer = config['regularizer']

        self._activation = config['activation']
        self._embed_size = config['embed_size']
        self._embed_dropout = config['embed_dropout']

    def get_data(self, train_file_dir):
        path_list = []
        file_list = []
        data_list = list()

        for root, dirs, files in os.walk(train_file_dir):
            for f in files:
                file_path = os.path.join(root, f)
                path_list.append(file_path)

        for i in range(len(path_list)):
            file_list.append(i)

        for i, path in enumerate(path_list):
            file_list[i] = read_csv(path, encoding="utf-8")

            if file_list[i].shape[0] > 15:
                data_list.append(file_list[i])

        return data_list

    def windows(self, dataset):
        datasets = []
        for set in dataset:
            for num in range(len(set)):
                if num + 14 < len(set):
                    batch_size = set[num: num + 15]
                    datasets.append(batch_size)
                else:
                    break

        datasets = np.array(datasets)

        return datasets

    def y_windows(self, dataset):
        datasets = []
        for set in dataset:
            for num in range(len(set)):
                if num + 14 < len(set):
                    batch_size = set[14 + num]
                    datasets.append(batch_size)
                else:
                    break

        datasets = np.array(datasets)

        return datasets

    def get_train_data(self, train_file_dir):

        data = self.get_data(train_file_dir)

        file_list = list()
        trainx_data = list()
        trainy_data = list()

        for datas in data:
            file = pd.DataFrame(data={
                'ped_coord': datas['ped_coord'].values,
                'speed': datas['speed'].values,
                'veh_speed': [1/theta for theta in datas['veh_speed'].values]
            })
            file_list.append(file)
        total_files = file_list[0]
        for i in range(len(file_list)):
            if i > 0:
                total_files = np.concatenate((total_files, file_list[i]), axis=0)

        x_max = np.array(total_files).max(axis=0)
        x_min = np.array(total_files).min(axis=0)
        for i in range(len(file_list)):
            file_list[i] = (file_list[i] - x_min) / (x_max - x_min)
            for j in range(len(file_list[i]['veh_speed'])):

                file_list[i]['veh_speed'][j] = 1 if file_list[i]['veh_speed'][j] <= 0.1 else 0

        for train in file_list:
            x_train = train.reindex(columns=['ped_coord', 'speed']).values
            trainy = train.reindex(columns=['veh_speed']).values

            trainx = x_train.astype('float32')
            trainy = trainy.astype('float32')

            trainx_data.append(trainx)
            trainy_data.append(trainy)

        x_train = self.windows(trainx_data)
        x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 2)).astype('float32')

        y_train = self.y_windows(trainy_data)
        y_train = y_train.reshape(y_train.shape[0], 1).astype('float32')

        x_train = np.array(x_train)
        y_train = np.array(y_train)

        x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, shuffle=True,
                                                            random_state=0)
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, shuffle=True,
                                                          random_state=0)
        train_data = (x_train, y_train)
        val_data = (x_val, y_val)
        test_data = (x_test, y_test)

        dataset = {
            'train': train_data,
            'val': val_data,
            'test': test_data
        }

        return dataset

    def train(self,
              train_data,
              val_data,
              batch_size=64,
              epochs=300,
              optimizer_type='rmsprop',
              optimizer_params={'lr': 0.0001, 'clipvalue': 0.0, 'decay': 0.0},
              loss=['binary_crossentropy'],
              metrics=['accuracy'],
              learning_scheduler=True,
              model_name=''):

        train_config = {'batch_size': batch_size,
                        'epoch': epochs,
                        'optimizer_type': optimizer_type,
                        'optimizer_params': optimizer_params,
                        'loss': loss,
                        'metrics': metrics,
                        'learning_scheduler_mode': 'plateau',
                        'learning_scheduler_params': {'exp_decay_param': 0.3,
                                                      'step_drop_rate': 0.5,
                                                      'epochs_drop_rate': 20.0,
                                                      'plateau_patience': 5,
                                                      'min_lr': 0.0000001,
                                                      'monitor_value': 'val_loss'},
                        'model': 'se_model',
                        'dataset': 'pie'}

        optimizer = RMSprop(lr=optimizer_params['lr'],
                            decay=optimizer_params['decay'],
                            clipvalue=optimizer_params['clipvalue'])

        model = self.get_model()

        model.compile(loss=loss, metrics=metrics, optimizer=optimizer)
        print('TRAINING: loss={} metrics={}'.format(loss, metrics))

        model_folder_name = 'se_model_pretrained'
        # model_folder_name = time.strftime("%d%b%Y-%Hh%Mm%Ss")

        model_path, _ = self.get_path(type_save='models',
                                      model_name='SE_Module',
                                      models_save_folder=model_folder_name,
                                      file_name='model.h5',
                                      save_root_folder='models')
        config_path, _ = self.get_path(type_save='models',
                                       model_name='SE_Module',
                                       models_save_folder=model_folder_name,
                                       file_name='configs',
                                       save_root_folder='models')

        with open(config_path + '.pkl', 'wb') as fid:
            pickle.dump([self.get_model_config(),
                         train_config],
                        fid, pickle.HIGHEST_PROTOCOL)
        print('Wrote configs to {}'.format(config_path))

        with open(config_path + '.txt', 'wt') as fid:
            fid.write("####### Data options #######\n")
            fid.write("\n####### Model config #######\n")
            fid.write(str(self.get_model_config()))
            fid.write("\n####### Training config #######\n")
            fid.write(str(train_config))

        print("##############################################")
        print(" Training for predicting sequences of size %d" % self._observe_length)
        print("##############################################")

        if os.path.exists(model_path) and os.path.getsize(model_path):
            print('检测到模型存在，正在加载模型..')
            model = load_model(model_path)

        checkpoint = ModelCheckpoint(filepath=model_path,
                                     save_best_only=True,
                                     save_weights_only=False,
                                     monitor=train_config['learning_scheduler_params']['monitor_value'])
        call_backs = [checkpoint]

        if learning_scheduler:
            early_stop = EarlyStopping(monitor='val_loss',
                                       min_delta=0.0001,
                                       patience=10,
                                       verbose=1)
            plateau_sch = ReduceLROnPlateau(train_config['learning_scheduler_params']['monitor_value'],
                                            factor=train_config['learning_scheduler_params']['step_drop_rate'],
                                            patience=train_config['learning_scheduler_params']['plateau_patience'],
                                            min_lr=train_config['learning_scheduler_params']['min_lr'],
                                            verbose=1)
            call_backs.extend([early_stop, plateau_sch])

        history = model.fit(x=train_data[0],
                            y=train_data[1],
                            batch_size=batch_size, epochs=epochs,
                            validation_data=val_data,
                            callbacks=call_backs,
                            verbose=1)

        history_path, saved_files_path = self.get_path(type_save='models',
                                                       model_name='se_model',
                                                       models_save_folder=model_folder_name,
                                                       file_name='history.pkl',
                                                       save_root_folder='models')

        print('Train trained_model is saved to {}'.format(model_path))

        with open(history_path, 'wb') as fid:
            pickle.dump(history.history, fid, pickle.HIGHEST_PROTOCOL)
        print('Wrote configs to {}'.format(config_path))

        del train_data, val_data

        return saved_files_path

    def test_chunk(self,
                   test_data,
                   model_path=''):
        with open(os.path.join(model_path, 'configs.pkl'), 'rb') as fid:
            try:
                configs = pickle.load(fid)
            except:
                configs = pickle.load(fid, encoding='bytes')
        train_params = configs[1]
        self.load_model_config(configs[0])
        self.se_model = self.get_model()

        try:
            test_model = load_model(os.path.join(model_path, 'model.h5'))

        except:
            test_model = self.get_model()
            test_model.load_weights(os.path.join(model_path, 'model.h5'))

        test_model.summary()

        test_target_data = []
        test_results = []

        for i in range(0, len(test_data[0]), 100):
            test_data_chunk = (test_data[0][i:i + 100], test_data[1][i:i + 100])

            test_results_chunk = test_model.predict(test_data_chunk[0],
                                                    batch_size=train_params['batch_size'],
                                                    verbose=1)

            test_target_data.extend(test_data_chunk[1])
            test_results.extend(test_results_chunk)

        acc = accuracy_score(test_target_data, np.round(test_results))
        f1 = f1_score(test_target_data, np.round(test_results))

        t = PrettyTable(['Acc', 'F1'])
        t.title = 'Speed-Estimation Model '
        t.add_row([acc, f1])

        print(t)

        save_results_path = os.path.join(model_path, 'speed_estimation.pkl')
        if not os.path.exists(save_results_path):
            results = {'results': test_results,
                       'groundtruth': test_target_data,
                       'accuracy': acc,
                       'f1 score': f1}
            with open(save_results_path, 'wb') as fid:
                pickle.dump(results, fid, pickle.HIGHEST_PROTOCOL)

        return test_results



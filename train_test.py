import os
import sys
import warnings
import argparse
import tensorflow as tf
import zipfile

from prettytable import PrettyTable
from os.path import join
from se_module import SE_Module
from loop_train import LoopTrain
from loop_predict import LoopPredict
from pie_data import PIE


def train_predict(dataset, data_path, train_test):

    loop_train = LoopTrain()
    loop_test = LoopPredict()

    pie_path = data_path

    if not train_test:
        loop_train.loop_train(data_path=pie_path)

    if train_test:
        perf_final = loop_test.loop_predict(data_path=pie_path)

        t = PrettyTable(['MSE', 'C_MSE'])
        t.title = 'Trajectory prediction model'
        t.add_row([perf_final['mse-45'], perf_final['c-mse-45']])

        print(t)


def pre_trained_SEModule(train_test):

    file_dir = os.path.dirname(__file__)
    path_train = join(file_dir, r'./data/pedestrian_view/')

    se_model = SE_Module(encoder_feature_size=2,
                         prediction_size=1,
                         observe_length=15,
                         num_hidden_units=64,
                         regularizer_val=0.001,
                         activation='softsign',
                         embed_size=64,
                         embed_dropout=0.35)
    data = se_model.get_train_data(path_train)

    if not train_test:  # Train
        se_model.train(data['train'],
                       data['val'],
                       batch_size=64,
                       epochs=300,
                       optimizer_type='rmsprop',
                       optimizer_params={'lr': 0.0001, 'clipvalue': 0.0, 'decay': 0.0},
                       loss=['binary_crossentropy'],
                       metrics=['accuracy'],
                       learning_scheduler=True,
                       model_name='se_model')

    if train_test:  # Test
        saved_files_path = join(file_dir, r'./models/SE_Module/se_model_pretrained')
        test_results = se_model.test_chunk(data['test'], model_path=saved_files_path)


def main(train_test, data_path):
    pre_trained_SEModule(train_test=train_test)
    train_predict(dataset='pie_dataset', data_path=data_path, train_test=train_test)


def unzip_data(data_path):
    annotations_path = os.path.join(data_path, 'annotations')
    annotations_zip = os.path.join(data_path, 'annotations.zip')
    annotations_attr_path = os.path.join(data_path, 'annotations_attributes')
    annotations_attr_path_zip = os.path.join(data_path, 'annotations_attributes.zip')
    annotations_veh_path = os.path.join(data_path, 'annotations_vehicle')
    annotations_veh_path_zip = os.path.join(data_path, 'annotations_vehicle.zip')

    dirs = [annotations_path, annotations_attr_path, annotations_veh_path]
    zips = [annotations_zip, annotations_attr_path_zip, annotations_veh_path_zip]

    def unzip_data(path_to_zip_file):
        with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
            zip_ref.extractall(data_path)

    for d, z in zip(dirs, zips):
        if not os.path.isdir(d):
            unzip_data(z)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='train and test',
    )
    parser.add_argument('--data', required=True, type=str)
    parser.add_argument('--test', required=False, action='store_true')

    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    #print(os.environ.copy()['DATA_PATH'])
    data_path = args.data
    train_test = args.test

    unzip_data(data_path)
    main(train_test=args.test, data_path=data_path)
    #try:
    #    main(train_test=args.test, data_path=data_path)
    #except ValueError:
    #    raise ValueError('Usage: python train_test.py <train_test>\n'
    #                     'train_test: 0 - train only, 1 - train and test, 2 - test only\n')

import copy
import os
import time
import pickle
import numpy as np
import warnings
import pie_data
import pp_module
import vp_module
import se_module

from tqdm import tqdm
from os.path import join, exists, getsize
from keras.layers.recurrent import LSTM
from keras.models import Model, load_model


class LoopTrain(object):

    def get_tracks(self, dataset, data_types, observe_length, predict_length, overlap, normalize):

        seq_length = observe_length + predict_length + 1
        overlap_stride = observe_length if overlap == 0 else \
            int((1 - overlap) * observe_length)
        overlap_stride = 1 if overlap_stride < 1 else overlap_stride

        d = {}
        for dt in data_types:
            try:
                d[dt] = dataset[dt]
            except KeyError:
                raise ('Wrong data type is selected %s' % dt)

        d['image'] = dataset['image']
        d['pid'] = dataset['pid']

        for k in d.keys():
            tracks = []
            for track in d[k]:
                tracks.extend([track[i:i + seq_length] for i in
                               range(0, len(track) - seq_length + 1, overlap_stride)])
            d[k] = tracks

        if normalize:
            if 'bbox' in data_types:
                for i in range(len(d['bbox'])):
                    d['bbox'][i] = np.subtract(d['bbox'][i][1:], d['bbox'][i][0]).tolist()
            if 'center' in data_types:
                for i in range(len(d['center'])):
                    d['center'][i] = np.subtract(d['center'][i][1:], d['center'][i][0]).tolist()
            if 'gps_coord' in data_types:
                for i in range(len(d['gps_coord'])):
                    d['gps_coord'][i] = np.subtract(d['gps_coord'][i][1:], d['gps_coord'][i][0]).tolist()
            if 'ped_coord' in data_types:
                x_max = np.array(d['ped_coord']).max(axis=0)
                x_min = np.array(d['ped_coord']).min(axis=0)
                for i in range(len(d['ped_coord'])):
                    d['ped_coord'][i] = (d['ped_coord'][i] - x_min) / (x_max - x_min)
            if 'speed' in data_types:
                s_max = np.array(d['speed']).max(axis=0)
                s_min = np.array(d['speed']).min(axis=0)
                for i in range(len(d['speed'])):
                    d['speed'][i] = (d['speed'][i] - s_min) / (s_max - s_min)

            for k in d.keys():
                if k != 'bbox' and k != 'center' and k != 'gps_coord':
                    for i in range(len(d[k])):
                        d[k][i] = d[k][i][1:]

        return d

    def get_data_helper(self, data, data_type):

        if not data_type:
            return []
        d = []
        for dt in data_type:
            if dt == 'image':
                continue
            d.append(np.array(data[dt]))

        if len(d) > 1:
            return np.concatenate(d, axis=2)
        else:
            return d[0]

    def short_windows(self, num, dataset):
        datasets = []
        for set in dataset:
            batch_size = set[num * 15: num * 15 + 15]
            datasets.append([batch_size, set[0:15]])
        datasets = np.array(datasets)

        return datasets

    def short_y_windows(self, num, dataset):
        datasets = []
        for set in dataset:
            batch_size = set[num * 15: num * 15 + 15]
            datasets.append(batch_size)
        datasets = np.array(datasets)

        return datasets

    def get_data(self, data, **model_opts):

        opts = {
            'normalize_bbox': True,
            'track_overlap': 0.5,
            'observe_length': 15,
            'predict_length': 45,
            'enc_input_type': ['bbox'],
            'dec_input_type': [],
            'prediction_type': ['bbox']
        }

        for key, value in model_opts.items():
            assert key in opts.keys(), 'wrong data parameter %s' % key
            opts[key] = value

        observe_length = opts['observe_length']
        data_types = set(opts['enc_input_type'] + opts['dec_input_type'] + opts['prediction_type'])
        data_tracks = self.get_tracks(data, data_types, observe_length,
                                      opts['predict_length'], opts['track_overlap'],
                                      opts['normalize_bbox'])

        if opts['normalize_bbox']:
            print('Normalization has been completed.')

        obs_slices = {}
        pred_slices = {}
        all_slices = {}

        for k in data_tracks.keys():
            obs_slices[k] = []
            pred_slices[k] = []
            all_slices[k] = []
            obs_slices[k].extend([d[0:observe_length] for d in data_tracks[k]])
            pred_slices[k].extend([d[observe_length:] for d in data_tracks[k]])
            all_slices[k].extend([d[0:] for d in data_tracks[k]])

        enc_input = self.get_data_helper(obs_slices, opts['enc_input_type'])
        enc_all_input = self.get_data_helper(all_slices, opts['enc_input_type'])

        dec_input = self.get_data_helper(pred_slices, opts['dec_input_type'])
        pred_target = self.get_data_helper(pred_slices, opts['prediction_type'])

        # if not len(dec_input) > 0:
        #     dec_input = np.zeros(shape=pred_target.shape)

        return {'obs_image': obs_slices['image'],
                'obs_pid': obs_slices['pid'],
                'pred_image': pred_slices['image'],
                'pred_pid': pred_slices['pid'],
                'enc_input': enc_all_input,
                'dec_input': dec_input,
                'pred_target': pred_target,
                'model_opts': opts}

    def speed_estimation(self, data, **model_opts):

        dec_input = np.zeros((data['enc_input'].shape[0], 15, 1))

        if 'ped_coord' in model_opts['enc_input_type']:
            model_folder_name = 'SE_Module'
            model_type = 'se_model_pretrained'
        else:
            model_folder_name = 'SE_Module'
            model_type = 'se_model_pretrained'
        print(model_type)

        se_model = se_module.SE_Module()
        _, model_path = se_model.get_path(type_save='models',
                                          model_name=model_folder_name,
                                          models_save_folder=model_type,
                                          file_name='model.h5',
                                          save_root_folder='models')

        with open(join(model_path, 'configs.pkl'), 'rb') as fid:
            try:
                configs = pickle.load(fid)
            except:
                configs = pickle.load(fid, encoding='bytes')
        train_params = configs[1]

        try:
            test_model = load_model(join(model_path, 'model.h5'))

        except:
            test_model = se_model.get_model()
            test_model.load_weights(join(model_path, 'model.h5'))

        print('Start speed estimation...')

        test_results = test_model.predict(data['enc_input'][:, :15],
                                          batch_size=train_params['batch_size'],
                                          verbose=0)

        for i in range(data['enc_input'].shape[0]):
            dec_input[i] = np.array([[test_results[i][0]]] * 15)

        print('Speed estimation is complete.')

        return dec_input

    def train(self, num,
              vp_model,
              vp_model_opts,
              pp_model,
              pp_model_opts,
              all_data):
        if num > 0:
            vp_train_data = {key: np.array([*all_data['obs_data'][0][key], *all_data['pred_data'][0][key]]) for key in
                             all_data['obs_data'][0]}
            vp_train_data['model_opts'] = all_data['obs_data'][0]['model_opts']

            vp_val_data = all_data['obs_data'][1]
            pp_train_data = {key: np.array([*all_data['obs_data'][2][key], *all_data['pred_data'][1][key]]) for key in
                             all_data['obs_data'][2]}
            pp_train_data['model_opts'] = all_data['obs_data'][2]['model_opts']
            pp_val_data = all_data['obs_data'][3]
        else:
            vp_train_data = all_data['obs_data'][0]
            vp_val_data = all_data['obs_data'][1]
            pp_train_data = all_data['obs_data'][2]
            pp_val_data = all_data['obs_data'][3]

        vp_saved_path = vp_model.train(num, vp_train_data, vp_val_data, **vp_model_opts)
        pp_saved_path = pp_model.train(num, pp_train_data, pp_val_data, **pp_model_opts)

        vp_testY, vp_testUn = vp_model.test(num, vp_train_data, 1, **vp_model_opts)
        pp_testY, pp_testUn = pp_model.test(num, pp_train_data, 1, **pp_model_opts)

        return vp_testY, vp_testUn, pp_testY, pp_testUn

    def loop_train(self, data_path='data/pie_dataset'):

        data_opts = {'fstride': 1,
                     'sample_type': 'all',
                     'height_rng': [0, float('inf')],
                     'squarify_ratio': 0,
                     'data_split_type': 'default',
                     'seq_type': 'crossing',
                     'min_track_size': 60,
                     'random_params': {'ratios': None,
                                       'val_data': True,
                                       'regen_data': True},
                     'kfold_params': {'num_folds': 5, 'fold': 1}}

        imdb = pie_data.PIE(data_path=data_path)

        beh_seq_val = imdb.generate_data_trajectory_sequence('val', **data_opts)
        beh_seq_train = imdb.generate_data_trajectory_sequence('train', **data_opts)

        se_model_opts = {'normalize_bbox': True,
                         'track_overlap': 0.5,
                         'observe_length': 15,
                         'predict_length': 45,
                         'enc_input_type': ['ped_coord', 'speed'],
                         'dec_input_type': [],
                         'prediction_type': []
                         }

        vp_model_opts = {'normalize_bbox': True,
                         'track_overlap': 0.5,
                         'observe_length': 15,
                         'predict_length': 45,
                         'enc_input_type': ['gps_coord', 'heading_angle', 'obd_speed', 'distance', 'bbox',
                                            'intention_prob', 'speed', 'uncertainty'],
                         'dec_input_type': ['speed_estimation'],
                         'prediction_type': ['gps_coord', 'heading_angle', 'obd_speed', 'distance']
                         }

        pp_model_opts = {'normalize_bbox': True,
                         'track_overlap': 0.5,
                         'observe_length': 15,
                         'predict_length': 45,
                         'enc_input_type': ['gps_coord', 'heading_angle', 'obd_speed', 'distance', 'bbox',
                                            'intention_prob', 'speed', 'uncertainty'],
                         'dec_input_type': ['speed_estimation'],
                         'prediction_type': ['bbox', 'intention_prob', 'speed']
                         }

        vp_model = vp_module.VP_Module()
        pp_model = pp_module.PP_Module()

        pretrained_train_data = self.get_data(beh_seq_train, **se_model_opts)
        pretrained_val_data = self.get_data(beh_seq_val, **se_model_opts)

        vp_train_data = vp_model.get_data(beh_seq_train, **vp_model_opts)
        vp_val_data = vp_model.get_data(beh_seq_val, **vp_model_opts)

        pp_train_data = pp_model.get_data(beh_seq_train, **pp_model_opts)
        pp_val_data = pp_model.get_data(beh_seq_val, **pp_model_opts)

        se_train_data = self.speed_estimation(pretrained_train_data, **se_model_opts)
        se_val_data = self.speed_estimation(pretrained_val_data, **se_model_opts)

        vp_train_data['dec_input'] = se_train_data
        vp_val_data['dec_input'] = se_val_data
        pp_train_data['dec_input'] = se_train_data
        pp_val_data['dec_input'] = se_val_data

        all_data = {
            'obs_data': ([copy.deepcopy(vp_train_data),
                          copy.deepcopy(vp_val_data),
                          copy.deepcopy(pp_train_data),
                          copy.deepcopy(pp_val_data)]),
            'pred_data': ([copy.deepcopy(vp_train_data),
                           copy.deepcopy(pp_train_data)])
        }

        num = 0

        while num < 3:
            all_data['obs_data'][0]['enc_input'] = self.short_windows(num, vp_train_data['enc_input'])
            all_data['obs_data'][0]['pred_target'] = self.short_y_windows(num, vp_train_data['pred_target'])

            all_data['obs_data'][1]['enc_input'] = self.short_windows(num, vp_val_data['enc_input'])
            all_data['obs_data'][1]['pred_target'] = self.short_y_windows(num, vp_val_data['pred_target'])

            all_data['obs_data'][2]['enc_input'] = self.short_windows(num, pp_train_data['enc_input'])
            all_data['obs_data'][2]['pred_target'] = self.short_y_windows(num, pp_train_data['pred_target'])

            all_data['obs_data'][3]['enc_input'] = self.short_windows(num, pp_val_data['enc_input'])
            all_data['obs_data'][3]['pred_target'] = self.short_y_windows(num, pp_val_data['pred_target'])

            print('第{}轮训练开始...'.format(num + 1))

            vp_testY, vp_testUn, pp_testY, pp_testUn = self.train(num,
                                                                  vp_model, vp_model_opts,
                                                                  pp_model, pp_model_opts,
                                                                  all_data)

            print('第{}轮训练完成！'.format(num + 1))

            num += 1
            un = np.concatenate((vp_testUn, pp_testUn), axis=2)
            un = np.reshape(np.mean(un, axis=-1), (un.shape[0], un.shape[1], 1))
            all_data['pred_data'][0]['enc_input'] = np.array(
                list(zip(np.concatenate((vp_testY, pp_testY, un), axis=2),
                         vp_train_data['enc_input'][:, 0:15, :])))
            all_data['pred_data'][0]['pred_target'] = self.short_y_windows(num, vp_train_data['pred_target'])

            all_data['pred_data'][1]['enc_input'] = np.array(
                list(zip(np.concatenate((vp_testY, pp_testY, un), axis=2),
                         pp_train_data['enc_input'][:, 0:15, :])))
            all_data['pred_data'][1]['pred_target'] = self.short_y_windows(num, pp_train_data['pred_target'])
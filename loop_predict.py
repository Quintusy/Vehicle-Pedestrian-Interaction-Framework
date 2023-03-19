import copy
import os
import time
import pickle
import numpy as np
import warnings
import pie_data
import pp_module
import se_module
import vp_module

from tqdm import tqdm
from os.path import join
from keras.layers.recurrent import LSTM
from keras.models import Model, load_model


class LoopPredict(object):

    def get_tracks(self, dataset, data_types, observe_length, predict_length, overlap, normalize):

        seq_length = observe_length + predict_length
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

    def get_path(self,
                 num,
                 file_name='',
                 save_folder='PP_Module',
                 model_type='pp_model',
                 save_root_folder='models/'):

        save_path = os.path.join(save_root_folder, save_folder, model_type)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        return os.path.join(save_path, str(num) + file_name), save_path

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

        print('en_input:', np.array(data['enc_input']).shape)

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

    def predict(self, num,
                vp_model,
                vp_model_opts,
                pp_model,
                pp_model_opts,
                test_data):
        vp_test_data = test_data['vp_data']
        pp_test_data = test_data['pp_data']

        vp_testY, vp_testUn = vp_model.test(num, vp_test_data, mc_times=50, **vp_model_opts)
        pp_testY, pp_testUn = pp_model.test(num, pp_test_data, mc_times=50, **pp_model_opts)

        return vp_testY, vp_testUn, pp_testY, pp_testUn

    def loop_predict(self, data_path='data/pie_dataset'):

        data_opts = {'fstride': 1,
                     'sample_type': 'all',
                     'height_rng': [0, float('inf')],
                     'squarify_ratio': 0,
                     'data_split_type': 'default',
                     'seq_type': 'trajectory',
                     'min_track_size': 60,
                     'random_params': {'ratios': None,
                                       'val_data': True,
                                       'regen_data': True},
                     'kfold_params': {'num_folds': 5, 'fold': 1}}


        imdb = pie_data.PIE(data_path=data_path)

        beh_seq_test = imdb.generate_data_trajectory_sequence('test', **data_opts)

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
                         'dec_input_type': ['intention_prob'],
                         'prediction_type': ['gps_coord', 'heading_angle', 'obd_speed', 'distance']
                         }

        pp_model_opts = {'normalize_bbox': True,
                         'track_overlap': 0.5,
                         'observe_length': 15,
                         'predict_length': 45,
                         'enc_input_type': ['gps_coord', 'heading_angle', 'obd_speed', 'distance', 'bbox',
                                            'intention_prob', 'speed', 'uncertainty'],
                         'dec_input_type': ['intention_prob'],
                         'prediction_type': ['bbox', 'intention_prob', 'speed']
                         }

        vp_model = vp_module.VP_Module()
        pp_model = pp_module.PP_Module()

        pretrained_test_data = self.get_data(beh_seq_test, **se_model_opts)

        vp_test_data = vp_model.get_data(beh_seq_test, **vp_model_opts)
        pp_test_data = pp_model.get_data(beh_seq_test, **pp_model_opts)

        se_test_data = self.speed_estimation(pretrained_test_data, **se_model_opts)

        vp_test_data['dec_input'] = se_test_data
        pp_test_data['dec_input'] = se_test_data

        model_folder_name = 'PP_Module'
        model_type = 'pp_model'

        test_data = {
            'vp_data': copy.deepcopy(vp_test_data),
            'pp_data': copy.deepcopy(pp_test_data)
        }

        traj = np.zeros((pp_test_data['pred_target'].shape[0], pp_test_data['pred_target'].shape[1], 4))
        print('The shape of traj is ：', traj.shape)

        num = 0

        test_data['vp_data']['enc_input'] = self.short_windows(num, vp_test_data['enc_input'])
        test_data['vp_data']['pred_target'] = self.short_y_windows(num, vp_test_data['pred_target'])

        test_data['pp_data']['enc_input'] = self.short_windows(num, pp_test_data['enc_input'])
        test_data['pp_data']['pred_target'] = self.short_y_windows(num, pp_test_data['pred_target'])

        while num < 3:
            print('Start the {}th round of predictions...'.format(num + 1))

            vp_testY, vp_testUn, pp_testY, pp_testUn = self.predict(num,
                                                                    vp_model, vp_model_opts,
                                                                    pp_model, pp_model_opts,
                                                                    test_data)

            print('The {}th round prediction completed！'.format(num + 1))

            traj[:, num * 15:num * 15 + 15, :] = pp_testY[:, :, :4]

            num += 1

            un = np.concatenate((vp_testUn, pp_testUn), axis=2)
            un = np.reshape(np.mean(un, axis=-1), (un.shape[0], un.shape[1], 1))
            test_data['vp_data']['enc_input'] = np.array(
                list(zip(np.concatenate((vp_testY, pp_testY, un), axis=2),
                         vp_test_data['enc_input'][:, 0:15, :])))
            test_data['vp_data']['pred_target'] = self.short_y_windows(num, vp_test_data['pred_target'])

            test_data['pp_data']['enc_input'] = np.array(
                list(zip(np.concatenate((vp_testY, pp_testY, un), axis=2),
                         pp_test_data['enc_input'][:, 0:15, :])))
            test_data['pp_data']['pred_target'] = self.short_y_windows(num, pp_test_data['pred_target'])

        perf = {}
        performance = np.square(pp_test_data['pred_target'][:, :, :4] - traj)
        perf['mse-15'] = performance[:, 0:15, :].mean(axis=None)
        perf['mse-30'] = performance[:, 0:30, :].mean(axis=None)
        perf['mse-45'] = performance.mean(axis=None)
        perf['mse-last'] = performance[:, -1, :].mean(axis=None)

        print("mse-15: %.2f\nmse-30: %.2f\nmse-45: %.2f"
              % (perf['mse-15'], perf['mse-30'], perf['mse-45']))
        print("mse-last %.2f\n" % (perf['mse-last']))

        res_centers = np.zeros(shape=(traj.shape[0], traj.shape[1], 2))
        centers = np.zeros(shape=(traj.shape[0], traj.shape[1], 2))
        for b in range(traj.shape[0]):
            for c in range(traj.shape[1]):
                centers[b, c, 0] = (pp_test_data['pred_target'][b, c, 2] + pp_test_data['pred_target'][b, c, 0]) / 2
                centers[b, c, 1] = (pp_test_data['pred_target'][b, c, 3] + pp_test_data['pred_target'][b, c, 1]) / 2
                res_centers[b, c, 0] = (traj[b, c, 2] + traj[b, c, 0]) / 2
                res_centers[b, c, 1] = (traj[b, c, 3] + traj[b, c, 1]) / 2

        c_performance = np.square(centers - res_centers)
        perf['c-mse-15'] = c_performance[:, 0:15, :].mean(axis=None)
        perf['c-mse-30'] = c_performance[:, 0:30, :].mean(axis=None)
        perf['c-mse-45'] = c_performance.mean(axis=None)
        perf['c-mse-last'] = c_performance[:, -1, :].mean(axis=None)

        print("c-mse-15: %.2f\nc-mse-30: %.2f\nc-mse-45: %.2f" \
              % (perf['c-mse-15'], perf['c-mse-30'], perf['c-mse-45']))
        print("c-mse-last: %.2f\n" % (perf['c-mse-last']))

        history_path, model_path = self.get_path(num,
                                                 save_folder=model_folder_name,
                                                 model_type=model_type,
                                                 file_name='history.pkl')

        save_results_path = join(model_path,
                                 '{:.2f}.pkl'.format(perf['mse-45']))
        save_performance_path = join(model_path,
                                     '{:.2f}.txt'.format(perf['mse-45']))

        with open(save_performance_path, 'wt') as fid:
            for k in sorted(perf.keys()):
                fid.write("%s: %s\n" % (k, str(perf[k])))

        if not os.path.exists(save_results_path):
            try:
                results = {'img_seqs': [],
                           'results': traj,
                           'gt': pp_test_data['pred_target'][:, :, :4],
                           'performance': perf}
            except:
                results = {'img_seqs': [],
                           'results': traj,
                           'gt': pp_test_data['pred_target'][:, :, :4],
                           'performance': perf}

            with open(save_results_path, 'wb') as fid:
                pickle.dump(results, fid, pickle.HIGHEST_PROTOCOL)

        return perf

if __name__ == '__main__':
    loop = LoopPredict()
    loop.loop_predict()
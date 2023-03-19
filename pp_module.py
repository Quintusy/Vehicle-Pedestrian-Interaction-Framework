import os
import time
import pickle
import numpy as np
import tensorflow as tf
import bi_layer_Bayesian_LSTM
import pie_data

from tqdm import tqdm
from os.path import join, exists, getsize
from keras.models import load_model
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.optimizers import RMSprop
from keras import regularizers


class PP_Module(object):

    def __init__(self,
                 num_hidden_units=256,
                 regularizer_val=0.0001,
                 activation='softsign',
                 embed_size=64,
                 embed_dropout=0):

        self._num_hidden_units = num_hidden_units
        self._regularizer_value = regularizer_val
        self._regularizer = regularizers.l2(regularizer_val)

        self._activation = activation
        self._embed_size = embed_size
        self._embed_dropout = embed_dropout

        self._observe_length = 15
        self._predict_length = 15

        self._encoder_feature_size = 11
        self._decoder_feature_size = 1

        self._prediction_size = 6

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

        if not len(dec_input) > 0:
            dec_input = np.zeros(shape=enc_input.shape)

        return {'obs_image': obs_slices['image'],
                'obs_pid': obs_slices['pid'],
                'pred_image': pred_slices['image'],
                'pred_pid': pred_slices['pid'],
                'enc_input': enc_all_input,
                'dec_input': dec_input,
                'pred_target': pred_target,
                'model_opts': opts}

    def short_windows(self, dataset):
        datasets = []
        for set in dataset:
            for num in range(3):
                batch_size = set[num * 15: num * 15 + 15]
                datasets.append([batch_size, set[0:15]])
        datasets = np.array(datasets)

        return datasets

    def short_y_windows(self, dataset):
        datasets = []
        for set in dataset:
            for num in range(3):
                batch_size = set[num * 15: num * 15 + 15]
                datasets.append(batch_size)
        datasets = np.array(datasets)

        return datasets

    def get_path(self, num,
                 file_name='',
                 save_folder='PP_Module',
                 model_type='pp_model',
                 save_root_folder='models/'):

        save_path = os.path.join(save_root_folder, save_folder, model_type)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        return os.path.join(save_path, str(num) + file_name), save_path

    def log_configs(self, config_path, batch_size, epochs,
                    lr, loss, learning_scheduler, opts):

        with open(config_path, 'wt') as fid:
            fid.write("####### Model options #######\n")
            for k in opts:
                fid.write("%s: %s\n" % (k, str(opts[k])))

            fid.write("\n####### Network config #######\n")
            fid.write("%s: %s\n" % ('hidden_units', str(self._num_hidden_units)))
            fid.write("%s: %s\n" % ('reg_value ', str(self._regularizer_value)))
            fid.write("%s: %s\n" % ('activation', str(self._activation)))
            fid.write("%s: %s\n" % ('embed_size', str(self._embed_size)))
            fid.write("%s: %s\n" % ('embed_dropout', str(self._embed_dropout)))

            fid.write("%s: %s\n" % ('observe_length', str(self._observe_length)))
            fid.write("%s: %s\n" % ('predict_length ', str(self._predict_length)))
            fid.write("%s: %s\n" % ('encoder_feature_size', str(self._encoder_feature_size)))
            fid.write("%s: %s\n" % ('decoder_feature_size', str(self._decoder_feature_size)))
            fid.write("%s: %s\n" % ('prediction_size', str(self._prediction_size)))

            fid.write("\n####### Training config #######\n")
            fid.write("%s: %s\n" % ('batch_size', str(batch_size)))
            fid.write("%s: %s\n" % ('epochs', str(epochs)))
            fid.write("%s: %s\n" % ('lr', str(lr)))
            fid.write("%s: %s\n" % ('loss', str(loss)))
            fid.write("%s: %s\n" % ('learning_scheduler', str(learning_scheduler)))

        print('Wrote configs to {}'.format(config_path))

    def loss_fuction(self, truth, pred):
        loss_function = 0
        for i in range(self._prediction_size):
            loss1 = tf.exp(-tf.clip_by_value(pred[:, :, self._prediction_size+i], 0, 5)) * tf.square(truth[:, :, i] - pred[:, :, i])
            loss2 = tf.clip_by_value(pred[:, :, self._prediction_size+i], 0, 5)
            loss_function += tf.reduce_mean(0.5 * loss1 + 0.5 * loss2)

        return loss_function

    def train(self, num,
              train_data, val_data,
              batch_size=256,
              epochs=10,
              lr=0.001,
              learning_scheduler=True,
              **model_opts):

        optimizer = RMSprop(lr=lr)
        loss = self.loss_fuction

        print("Number of samples:\n Train: %d \n Val: %d \n"
              % (train_data['enc_input'].shape[0], val_data['enc_input'].shape[0]))

        self._observe_length = train_data['enc_input'].shape[1]
        self._predict_length = train_data['pred_target'].shape[1]

        self._encoder_feature_size = train_data['enc_input'].shape[3]
        self._decoder_feature_size = train_data['dec_input'].shape[2]

        self._prediction_size = train_data['pred_target'].shape[2]

        # Set path names for saving configs and model
        # model_folder_name = time.strftime("%d%b%Y-%Hh%Mm%Ss")

        if 'bbox' in model_opts['prediction_type']:
            model_folder_name = 'PP_Module'
            model_type = 'pp_model'
        else:
            model_folder_name = 'VP_Module'
            model_type = 'vp_model'
        print(model_type)

        model_path, _ = self.get_path(num,
                                      save_folder=model_folder_name,
                                      model_type=model_type,
                                      file_name='model.h5')

        opts_path, _ = self.get_path(num,
                                     save_folder=model_folder_name,
                                     model_type=model_type,
                                     file_name='model_opts.pkl')

        with open(opts_path, 'wb') as fid:
            pickle.dump(train_data['model_opts'], fid,
                        pickle.HIGHEST_PROTOCOL)

        config_path, _ = self.get_path(num,
                                       save_folder=model_folder_name,
                                       model_type=model_type,
                                       file_name='configs.txt')
        self.log_configs(config_path, batch_size, epochs,
                         lr, loss, learning_scheduler,
                         train_data['model_opts'])

        if exists(model_path) and getsize(model_path):
            pp_model = load_model(model_path, custom_objects={'loss_fuction': loss})
        else:
            pp_model = self.pp_model()
            # pp_model.load_weights(model_path)

        train_data = ([train_data['enc_input'][:100, 0],
                       train_data['enc_input'][:100, 1],
                       train_data['dec_input'][:100]],
                      train_data['pred_target'][:100])

        val_data = ([val_data['enc_input'][:100, 0],
                     val_data['enc_input'][:100, 1],
                     val_data['dec_input'][:100]],
                    val_data['pred_target'][:100])

        pp_model.compile(loss=loss, optimizer=optimizer)

        print("##############################################")
        print(" Training for predicting sequences of size %d" % self._predict_length)
        print("##############################################")

        checkpoint = ModelCheckpoint(filepath=model_path,
                                     save_best_only=True,
                                     save_weights_only=False,
                                     monitor='val_loss')
        call_backs = [checkpoint]

        if learning_scheduler:
            early_stop = EarlyStopping(monitor='val_loss',
                                       min_delta=1.0, patience=10,
                                       verbose=1)
            plateau_sch = ReduceLROnPlateau(monitor='val_loss',
                                            factor=0.2, patience=5,
                                            min_lr=1e-07, verbose=1)
            call_backs.extend([early_stop, plateau_sch])

        history = pp_model.fit(x=train_data[0], y=train_data[1],
                               batch_size=batch_size, epochs=epochs,
                               validation_data=val_data, verbose=1,
                               callbacks=call_backs)

        print('Train model is saved to {}'.format(model_path))

        history_path, saved_files_path = self.get_path(num,
                                                       save_folder=model_folder_name,
                                                       model_type=model_type,
                                                       file_name='history.pkl')

        with open(history_path, 'wb') as fid:
            pickle.dump(history.history, fid, pickle.HIGHEST_PROTOCOL)

        return saved_files_path

    def test(self, num, test_data, mc_times=100, **model_opts):

        if 'bbox' in model_opts['prediction_type']:
            model_folder_name = 'PP_Module'
            model_type = 'pp_model'
        else:
            model_folder_name = 'VP_Module'
            model_type = 'vp_model'
        print(model_type)

        history_path, model_path = self.get_path(num,
                                                 save_folder=model_folder_name,
                                                 model_type=model_type,
                                                 file_name='history.pkl')

        test_model = load_model(os.path.join(model_path, str(num) + 'model.h5'),
                                custom_objects={'loss_fuction': self.loss_fuction})
        test_model.summary()

        with open(os.path.join(model_path, str(num) + 'model_opts.pkl'), 'rb') as fid:
            try:
                model_opts = pickle.load(fid)
            except:
                model_opts = pickle.load(fid, encoding='bytes')

        test_obs_data = ([test_data['enc_input'][:, 0],
                          test_data['enc_input'][:, 1],
                          test_data['dec_input']])
        test_target_data = test_data['pred_target']

        pred = list()
        au = list()

        try:
            with tqdm(range(mc_times), desc='Monte Carlo Sampling') as tqdm_range:
                for t in tqdm_range:
                    results = test_model.predict(test_obs_data, batch_size=2048, verbose=0)
                    pred.append(results[:, :, :6])
                    au.append(results[:, :, 6:])
        except KeyboardInterrupt:
            tqdm_range.close()
        tqdm_range.close()

        e_u = np.var(np.array(pred), axis=0)
        a_u = np.exp(np.mean(np.array(au), axis=0))
        testUn = e_u + a_u
        testUn = 1 / (1 + np.exp(-np.sqrt(testUn)))

        testY = np.mean(np.array(pred), axis=0)
        np.square(test_target_data[:, :, :6] - testY)

        return testY, testUn

    def pp_model(self):

        bi_model = bi_layer_Bayesian_LSTM.bi_layer_Bayesian_LSTM(observe_length=15,
                                                                 encoder_feature_size=11,
                                                                 decoder_feature_size=1,
                                                                 predict_length=15,
                                                                 prediction_size=6, )
        model = bi_model.model()

        return model

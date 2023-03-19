from keras.layers import Input
from keras.layers import RepeatVector
from keras.layers import Dense
from keras.layers import Permute
from keras.layers import Concatenate
from keras.layers import Multiply
from keras.layers import Dropout
from keras.layers.recurrent import LSTM
from keras.layers import subtract
from keras.models import Model
from keras import regularizers


class bi_layer_Bayesian_LSTM(object):

    def __init__(self,
                 observe_length=15,
                 encoder_feature_size=11,
                 decoder_feature_size=1,
                 predict_length=15,
                 prediction_size=6,
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

        self._observe_length = observe_length
        self._predict_length = predict_length

        self._encoder_feature_size = encoder_feature_size
        self._decoder_feature_size = decoder_feature_size

        self._prediction_size = prediction_size

    def model(self):
        xi_encoder_input = Input(shape=(self._observe_length, self._encoder_feature_size),
                                 name='xi_encoder_input')
        xi_attention_net = self.attention_temporal(xi_encoder_input, self._observe_length)

        xi_encoder_model = self.create_lstm_model(name='xi_encoder_network')
        xi_encoder_outputs_states = xi_encoder_model(xi_attention_net)
        xi_encoder_states = xi_encoder_outputs_states[1:]

        decoder_model = self.create_lstm_model(name='decoder_network', r_state=False)
        xi_hidden_input = RepeatVector(self._predict_length)(xi_encoder_states[0])
        xi_embedded_hidden_input = Dense(self._embed_size, activation='relu')(xi_hidden_input)

        xo_encoder_input = Input(shape=(self._observe_length, self._encoder_feature_size),
                                 name='xo_encoder_input')
        xo_sub_input = subtract([xi_encoder_input, xo_encoder_input])
        xo_attention_net = self.attention_temporal(xo_sub_input, self._observe_length)
        xo_encoder_model = self.create_lstm_model(name='xo_encoder_network')
        xo_encoder_outputs_states = xo_encoder_model(xo_attention_net)
        xo_encoder_states = xo_encoder_outputs_states[1:]
        xo_hidden_input = RepeatVector(self._predict_length)(xo_encoder_states[0])
        xo_embedded_hidden_input = Dense(self._embed_size, activation='relu')(xo_hidden_input)

        _decoder_input = Input(shape=(self._predict_length, self._decoder_feature_size),
                               name='pred_decoder_input')
        decoder_concat_inputs = Concatenate(axis=2)(
            [xi_embedded_hidden_input, xo_embedded_hidden_input, _decoder_input])

        att_input_dim = self._embed_size + self._embed_size + self._decoder_feature_size
        decoder_concat_inputs = self.attention_element(decoder_concat_inputs, att_input_dim)

        decoder_output = decoder_model(decoder_concat_inputs,
                                       initial_state=xi_encoder_states)

        dropout_output = Dropout(self._embed_dropout, name='dropout_dec_output')(decoder_output)

        pred_y = Dense(self._prediction_size,
                       activation='linear',
                       name='pred_y')(dropout_output)

        log_var = Dense(self._prediction_size,
                        activation='linear',
                        name='log_var')(dropout_output)

        pred = Concatenate(axis=2)([pred_y, log_var])

        net_model = Model(inputs=[xi_encoder_input, xo_encoder_input, _decoder_input],
                          outputs=pred)

        net_model.summary()

        return net_model

    def create_lstm_model(self, name='lstm', r_state=True, r_sequence=True):
        network = LSTM(units=self._num_hidden_units,
                       return_state=r_state,
                       return_sequences=r_sequence,
                       stateful=False,
                       kernel_regularizer=self._regularizer,
                       recurrent_regularizer=self._regularizer,
                       bias_regularizer=self._regularizer,
                       activity_regularizer=None,
                       activation=self._activation,
                       name=name)

        return network

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

import tensorflow as tf
from tensorflow.contrib.slim.nets import resnet_v1
from tensorflow.contrib.rnn import LSTMCell
slim = tf.contrib.slim

from wrappers import Wrapper

class network(object):

    def __init__(self, FLAGS, cfg, mode):

        self.bookkeeping = {}
        self.plh = {}
        self.var = {}
        self.networks = self.get_network_types()

        self.FLAGS = FLAGS
        self.mode = mode
        self.cfg = cfg
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
    
    def get_network_types(self):

        networks = {}

        networks['appearance_networks'] = [
            'appearance_blstm_mtp',
            'appearance_blstm'
        ]

        networks['motion_networks'] = [
            'motion_lstm'
        ]

        networks['appearance_motion_networks'] = [
            'appearance_motion_network'          
        ]

        return networks

    def init_weights(self, name, weight_decay, shape):
        init = tf.random_normal_initializer(mean=0.0,stddev=0.01)
        regl = slim.l2_regularizer(weight_decay)
        return slim.variable(name, shape, initializer=init, regularizer=regl)

    def init_biases(self, name, shape):
        return slim.variable(name, shape, initializer=tf.constant_initializer(0.0))

    def compute_loss(self, predictions):

        detbb_num = self.var['detbb_num']
        track_num = self.var['track_num']
        sel_indices = self.plh['sel_indices']

        with tf.variable_scope('cross_entropy_loss'):
            # cls_prediction: N2 x N1 x T x output_dim, 
            # cls_prediction_vec: (N2 * N1 * T) x output_dim
            cls_prediction, cls_prediction_vec = predictions
            # (N2 * N1 * T) x output_dim
            softmax_out = tf.nn.softmax(cls_prediction_vec)
            # (N2 * N1 * T) x 1
            predicted_classes = tf.cast(tf.argmax(softmax_out, axis=1), tf.int32)

            # create labels
            idx1 = tf.range(detbb_num)
            idx2 = tf.range(track_num)
            idx1 = tf.expand_dims(idx1, axis=0)
            idx2 = tf.expand_dims(idx2, axis=1)
            # N2 x N1
            idx1 = tf.tile(idx1, [track_num, 1])
            idx2 = tf.tile(idx2, [1, detbb_num])

            # N2 x N1 x T
            labels = self.create_labels_for_matching(
                         idx1,
                         idx2,
                         tf.shape(cls_prediction)[2],
                         is_flip=False
                     )
            
            # (N2 * N1 * T) x 1
            labels_vec = tf.reshape(labels, [-1])

            weights = tf.ones_like(labels, dtype=tf.float32)
            
            # Focal Loss
            with tf.control_dependencies([tf.assert_less_equal(labels_vec, 1)]): # labels must be binary
                # create focal loss weights 
                p_t = tf.diag_part(tf.gather(softmax_out, labels_vec, axis=1))
                # (N2 * N1 * T)
                focal_weight = tf.square(1 - p_t)
                # N2 x N1 x T
                focal_weight = tf.reshape(focal_weight, [track_num, detbb_num, -1])
            # N2 x N1 x T
            weights = tf.multiply(weights, focal_weight)
            
            # create flipped labels
            inverted_labels = self.create_labels_for_matching(
                                    idx1,
                                    idx2,
                                    tf.shape(cls_prediction)[2],
                                    is_flip=True
                                )
            
            # additional weights for handling class imbalance
            pos_weight = 4.0
            neg_weight = 1.0

            cls_imb_weight = tf.multiply(pos_weight, tf.cast(labels, tf.float32)) + \
                                tf.multiply(neg_weight, tf.cast(inverted_labels, tf.float32))
            # N2 x N1 x T
            weights = tf.multiply(weights, cls_imb_weight)
            
            # apply weights
            weights = tf.multiply(weights, sel_indices)

            # N2 x N1 x T
            # do not add the loss to the colection yet
            loss = tf.losses.sparse_softmax_cross_entropy(
                       labels=labels,
                       logits=cls_prediction,
                       weights=weights,
                       loss_collection=None,
                       reduction=tf.losses.Reduction.NONE
                   )

            # N2 x N1
            loss = tf.reduce_sum(loss, axis=-1)  

            assert(self.cfg.SAMPLE_NUM_FOR_LOSS == 1)

            if self.FLAGS.network_type in self.networks['appearance_networks'] or \
               self.FLAGS.network_type in self.networks['appearance_motion_networks']:
                boundaries = [60168*2]
                # boundaries = [17696*2]
                values = [1000000, 30]
            elif self.FLAGS.network_type in self.networks['motion_networks']:
                boundaries = [60168*1]
                # boundaries = [17696*1]
                values = [1000000, 30]
            else:
                raise NotImplementedError
            top_k_num = tf.compat.v1.train.piecewise_constant(self.global_step, boundaries, values)

            # hard example mining 
            loss_tmp = tf.reshape(loss, [-1]) 
            top_k = tf.math.minimum(top_k_num, tf.shape(loss_tmp)[0])
            loss, _ = tf.math.top_k(loss_tmp, top_k, sorted=False)

            # scalar
            loss = tf.reduce_mean(loss, axis=None)        
            # add loss to the collection now
            tf.losses.add_loss(loss, loss_collection=tf.GraphKeys.LOSSES)

            reg_loss = tf.losses.get_regularization_loss()
            total_loss = tf.losses.get_total_loss()
            
            sel_flat = tf.cast(tf.reshape(sel_indices, [-1]), tf.bool)
            predicted_classes_tmp = tf.boolean_mask(predicted_classes, sel_flat)
            labels_vec_tmp = tf.boolean_mask(labels_vec, sel_flat)

            check_output = tf.cast(tf.equal(predicted_classes_tmp, labels_vec_tmp), tf.float32)
            accuracy_total = tf.reduce_sum(check_output)/tf.cast(tf.shape(check_output)[0], tf.float32)

            pos_label_idx = tf.equal(labels_vec_tmp, 1)
            neg_label_idx = tf.equal(labels_vec_tmp, 0)
            tp = tf.reduce_sum(tf.cast(
                     tf.equal(tf.boolean_mask(predicted_classes_tmp, pos_label_idx), 1), tf.float32))
            fp = tf.reduce_sum(
                     tf.cast(tf.equal(tf.boolean_mask(predicted_classes_tmp, neg_label_idx), 1), tf.float32))        
            fn = tf.reduce_sum(
                     tf.cast(tf.equal(tf.boolean_mask(predicted_classes_tmp, pos_label_idx), 0), tf.float32))
            tn = tf.reduce_sum(
                     tf.cast(tf.equal(tf.boolean_mask(predicted_classes_tmp, neg_label_idx), 0), tf.float32))

            recall = tp / (tp + fn)
            precision = tp / (tp + fp)
            f1 = 2 * (recall * precision) / (recall + precision)

            losses = total_loss, reg_loss, loss

        self.bookkeeping['softmax'] = softmax_out
        self.bookkeeping['predicted_classes'] = predicted_classes
        self.bookkeeping['labels_vec'] = labels_vec
        self.bookkeeping['total_loss'] = total_loss
        self.bookkeeping['reg_loss'] = reg_loss
        self.bookkeeping['loss'] = loss
        self.bookkeeping['accuracy_total'] = accuracy_total
        self.bookkeeping['recall'] = recall
        self.bookkeeping['precision'] = precision
        self.bookkeeping['f1'] = f1
        self.bookkeeping['tp'] = tp
        self.bookkeeping['fp'] = fp
        self.bookkeeping['tn'] = tn
        self.bookkeeping['fn'] = fn

        return losses

    def get_train_op(self, total_loss):

        if self.FLAGS.network_type in self.networks['appearance_networks']:
            train_op = self.get_appnet_train_op(total_loss)
        elif self.FLAGS.network_type in self.networks['motion_networks']:
            train_op = self.get_motnet_train_op(total_loss)
        elif self.FLAGS.network_type in self.networks['appearance_motion_networks']:
            train_op = self.get_appmotnet_train_op(total_loss)
        else:
            raise NotImplementedError

        return train_op

    def get_appnet_train_op(self, total_loss):
        
        with tf.variable_scope('training'):
            # boundaries = [17696*4, 17696*8] # MOT17 all  
            # values = [0.005, 0.0005, 0.00005] 
            boundaries = [60168*4, 60168*8] # MOT17 all  
            values = [0.005, 0.0005, 0.00005] 

            # NOTE: Note that global_step needs to be reinitialized or the boundaries need to be properly set,
            # if the model is restored from a saver object (instead of cnn_saver, app_lstm_saver, or mot_lstm_saver).
            # This is because global_step will be restored from the checkpoint file as well.
            learning_rate = tf.compat.v1.train.piecewise_constant(
                                self.global_step, boundaries, values)

            if self.FLAGS.finetuning == True:
                early_layers = self.get_layerwise_vars_to_train(
                                   layer_patterns=['block3', 'block4'],
                                   exclude_patterns=['BatchNorm'])
            elif self.FLAGS.finetuning == False:
                early_layers = self.get_layerwise_vars_to_train(
                                   layer_patterns=['block1', 'block2', 'block3', 'block4'],
                                   exclude_patterns=['BatchNorm'])
            else:
                raise NotImplementedError
            last_layers = self.get_layerwise_vars_to_train(
                              layer_patterns=['app_lstm'],
                              exclude_patterns=['BatchNorm'])
            train_op = self.build_train_op(
                           total_loss,
                           self.global_step,
                           learning_rate,
                           early_layers,
                           last_layers
                       )

            self.print_layer_vars(early_layers, layer_type='Early layer')
            self.print_layer_vars(last_layers, layer_type='Last layer')
            self.bookkeeping['lr'] = learning_rate
        
        return train_op

    def get_motnet_train_op(self, total_loss):

        with tf.variable_scope('training'):
            boundaries = [60168*2, 60168*3] # all sequences seqlen 10
            values = [0.001, 0.0001, 0.00001]
            # boundaries = [17696*2, 17696*3] # all sequences seqlen 10
            # values = [0.001, 0.0001, 0.00001]

            learning_rate = tf.compat.v1.train.piecewise_constant(
                                self.global_step, boundaries, values)

            variables_to_train = self.get_variables_to_train()
            optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(
                           total_loss,
                           global_step=self.global_step,
                           var_list=variables_to_train)
            
            self.bookkeeping['lr'] = learning_rate

        return train_op

    def get_appmotnet_train_op(self, total_loss):

        with tf.variable_scope('training'):
            # boundaries = [17696*2, 17696*2*2] # all training sequences
            # values = [0.005, 0.0005, 0.00005]
            boundaries = [60168*2, 60168*2*2] # all training sequences
            values = [0.005, 0.0005, 0.00005]

            learning_rate = tf.compat.v1.train.piecewise_constant(
                                self.global_step, boundaries, values)

            if self.FLAGS.finetuning == True:
                early_layers = self.get_layerwise_vars_to_train(
                                   layer_patterns=['app_lstm', 'mot_lstm'],
                                   exclude_patterns=['BatchNorm'])
            else:
                raise NotImplementedError
            
            last_layers = self.get_layerwise_vars_to_train(
                              layer_patterns=['app_mot_predictor'],
                              exclude_patterns=['BatchNorm'])
            
            train_op = self.build_train_op(
                           total_loss,
                           self.global_step,
                           learning_rate,
                           early_layers,
                           last_layers
                       )
            
            self.print_layer_vars(early_layers, layer_type='Early layer')
            self.print_layer_vars(last_layers, layer_type='Last layer')
            self.bookkeeping['lr'] = learning_rate

        return train_op

    def get_logging_op(self):

        bookkeeping = self.bookkeeping

        with tf.variable_scope('logging'):
        
            train_log_ops = {}
            val_log_ops = {}
            if self.mode == 'train':
                train_cls_loss_summary = tf.summary.scalar('training_cls_loss', bookkeeping['loss'])
                val_cls_loss_summary = tf.summary.scalar('validation_cls_loss', bookkeeping['loss'])
                train_total_loss_summary = tf.summary.scalar('training_total_loss', bookkeeping['total_loss'])
                val_total_loss_summary = tf.summary.scalar('validation_total_loss', bookkeeping['total_loss'])
                train_acc_loss_summary = tf.summary.scalar('training_accuracy', bookkeeping['accuracy_total'])
                val_acc_loss_summary = tf.summary.scalar('validation_accuracy', bookkeeping['accuracy_total'])
                train_rec_summary = tf.summary.scalar('training_recall', bookkeeping['recall'])
                val_rec_summary = tf.summary.scalar('validation_recall', bookkeeping['recall'])
                train_pre_summary = tf.summary.scalar('training_precision', bookkeeping['precision'])
                val_pre_summary = tf.summary.scalar('validation_precision', bookkeeping['precision'])
                train_f1_summary = tf.summary.scalar('training_f1', bookkeeping['f1'])
                val_f1_summary = tf.summary.scalar('validation_f1', bookkeeping['f1'])
                lr_summary = tf.summary.scalar('learning_rate', bookkeeping['lr'])

                proto_buffer = []
                for trainable_var in tf.trainable_variables():
                    proto_buffer.append(tf.summary.histogram(trainable_var.name, trainable_var))
                merged = tf.summary.merge(proto_buffer)

                train_log_ops['merged'] = merged
                train_log_ops['train_cls_loss_summary'] = train_cls_loss_summary
                train_log_ops['train_total_loss_summary'] = train_total_loss_summary
                train_log_ops['train_acc_loss_summary'] = train_acc_loss_summary
                train_log_ops['train_rec_summary'] = train_rec_summary
                train_log_ops['train_pre_summary'] = train_pre_summary
                train_log_ops['train_f1_summary'] = train_f1_summary
                train_log_ops['lr_summary'] = lr_summary

                val_log_ops['val_cls_loss_summary'] = val_cls_loss_summary
                val_log_ops['val_total_loss_summary'] = val_total_loss_summary
                val_log_ops['val_acc_loss_summary'] = val_acc_loss_summary
                val_log_ops['val_rec_summary'] = val_rec_summary
                val_log_ops['val_pre_summary'] = val_pre_summary
                val_log_ops['val_f1_summary'] = val_f1_summary

            saver = tf.train.Saver(max_to_keep=100)

            savers = {}
            if self.FLAGS.network_type in self.networks['appearance_networks']:
                cnn_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='resnet_v1_50')
                app_lstm_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='app_lstm')
                app_lstm_embd_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='app_lstm/app_embedding')
                app_lstm_lstm_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='app_lstm/lstm')

                cnn_saver = tf.train.Saver(cnn_vars)                   
                cnn_lstm_saver = tf.train.Saver(cnn_vars + app_lstm_vars)
                cnn_lstm_wopred_saver = tf.train.Saver(cnn_vars + app_lstm_embd_vars + app_lstm_lstm_vars)
                
                savers['saver'] = saver
                savers['cnn_saver'] = cnn_saver
                savers['cnn_lstm_saver'] = cnn_lstm_saver
                savers['cnn_lstm_wopred_saver'] = cnn_lstm_wopred_saver

            elif self.FLAGS.network_type in self.networks['motion_networks']:
                mot_lstm_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='mot_lstm')
                mot_lstm_saver = tf.train.Saver(mot_lstm_vars)

                savers['saver'] = saver
                savers['mot_lstm_saver'] = mot_lstm_saver

            elif self.FLAGS.network_type in self.networks['appearance_motion_networks']:
                cnn_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='resnet_v1_50')
                app_lstm_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='app_lstm')
                app_lstm_embd_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='app_lstm/app_embedding')
                app_lstm_lstm_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='app_lstm/lstm')
                mot_lstm_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='mot_lstm')
                app_mot_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='app_mot_predictor')

                cnn_saver = tf.train.Saver(cnn_vars)                   
                cnn_lstm_saver = tf.train.Saver(cnn_vars + app_lstm_vars)
                mot_lstm_saver = tf.train.Saver(mot_lstm_vars)
                cnn_lstm_wopred_saver = tf.train.Saver(cnn_vars + app_lstm_embd_vars + app_lstm_lstm_vars)
                cnn_app_mot_lstm_saver = tf.train.Saver(cnn_vars + app_lstm_vars + mot_lstm_vars + app_mot_vars)

                savers['saver'] = saver
                savers['cnn_saver'] = cnn_saver
                savers['cnn_lstm_saver'] = cnn_lstm_saver
                savers['mot_lstm_saver'] = mot_lstm_saver
                savers['cnn_lstm_wopred_saver'] = cnn_lstm_wopred_saver
                savers['cnn_app_mot_lstm_saver'] = cnn_app_mot_lstm_saver

            else:
                raise NotImplementedError

        logging_ops = {}
        logging_ops['savers'] = savers
        logging_ops['train_log_ops'] = train_log_ops
        logging_ops['val_log_ops'] = val_log_ops

        return logging_ops

    def get_variables_to_train(self):

        return tf.trainable_variables()
        
    def get_layerwise_vars_to_train(self, layer_patterns, exclude_patterns):

        layer_vars = []

        for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
            is_exclude_pattern_found = False
            for exclude_pattern in exclude_patterns:            
                if exclude_pattern in var.name:
                    is_exclude_pattern_found = True
                    break

            if is_exclude_pattern_found is True:
                continue

            for layer_pattern in layer_patterns:
                if layer_pattern in var.name:
                    layer_vars.append(var)
                    break

        return layer_vars

    def build_train_op(self, loss, global_step, learning_rate, early_layers, last_layers):

        opt1 = tf.train.GradientDescentOptimizer(0.1*learning_rate)
        opt2 = tf.train.GradientDescentOptimizer(learning_rate)
        grads = tf.gradients(loss, early_layers + last_layers)
        grads1 = grads[:len(early_layers)]
        grads2 = grads[len(early_layers):]
        train_op1 = opt1.apply_gradients(zip(grads1, early_layers), global_step=global_step)
        train_op2 = opt2.apply_gradients(zip(grads2, last_layers), global_step=global_step)    
        train_op = tf.group(train_op1, train_op2)

        return train_op

    def print_layer_vars(self, layer_vars, layer_type):

        for layer_var in layer_vars:
            print(layer_type + ": " + layer_var.name)

    def create_labels_for_matching(self, idx1, idx2, time_dimension, is_flip=False):
        
        if is_flip == False:
            # N2 x N1
            labels = tf.cast(tf.equal(idx1, idx2), tf.int32)
        else:
            labels = tf.cast(tf.logical_not(tf.equal(idx1, idx2)), tf.int32)
        # N2 x N1 x 1
        labels = tf.expand_dims(labels, axis=2)
        # N2 x N1 x T
        labels = tf.tile(labels, [1, 1, time_dimension])

        return labels

    def align_with_frames(self, h_app_states, track_len, num_step):

        def map_op(args):
            # arg1: (MAX_STEP, d), arg2: (1, ), arg3: (1, )
            arg1, arg2, arg3 = args
            arg3 = tf.reshape(arg3, [])
            with tf.control_dependencies([tf.assert_equal(tf.reduce_sum(arg1[arg3:, :]), 0.0)]):
                shifted_res = tf.manip.roll(input=arg1, shift=arg2, axis=[0])
            return shifted_res

        # (N2, )
        shift = num_step - track_len
        # (N2, 1)
        shift = tf.reshape(shift, [-1, 1])
        track_len = tf.reshape(track_len, [-1, 1])

        # align different rows of h_app_states by using the shift variable 
        # h_app_states: (N2, MAX_STEP, d), shift: (N2, 1)
        aligned_states = tf.map_fn(map_op, (h_app_states, shift, track_len), dtype=tf.float32)
        
        return aligned_states

    def align_with_frames_no_assert(self, h_app_states, track_len, num_step):
        "same as align_with_frames except for assert statement"

        def map_op(args):
            # arg1: (MAX_STEP, d), arg2: (1, ), arg3: (1, )
            arg1, arg2, arg3 = args
            arg3 = tf.reshape(arg3, [])
            shifted_res = tf.manip.roll(input=arg1, shift=arg2, axis=[0])
            return shifted_res

        # (N2, )
        shift = num_step - track_len
        # (N2, 1)
        shift = tf.reshape(shift, [-1, 1])
        track_len = tf.reshape(track_len, [-1, 1])

        # align different rows of h_app_states by using the shift variable  
        # h_app_states: (N2, MAX_STEP, d), shift: (N2, 1)
        aligned_states = tf.map_fn(map_op, (h_app_states, shift, track_len), dtype=tf.float32)
        
        return aligned_states

    def align_by_start_offset(self, h_mot_states, s_offset):

        def map_op(args):
            # arg1: (MAX_STEP, d), arg2: (1, ), arg3: (1, )
            arg1, arg2 = args    
            # with tf.control_dependencies([tf.assert_equal(tf.reduce_sum(arg1[:arg2, :]), 0.0)]):
            shifted_res = tf.manip.roll(input=arg1, shift=arg2, axis=[0])
            return shifted_res

        s_offset = tf.reshape(s_offset, [-1, 1])

        with tf.control_dependencies(
            [tf.assert_equal(tf.shape(s_offset)[0], tf.shape(h_mot_states)[0])]
        ):
            aligned_states = tf.map_fn(map_op, (h_mot_states, s_offset), dtype=tf.float32)
        
        return aligned_states

    def gather_final_states(self, c_app_states_all, track_len):

        def map_op(args):
            # arg1: (MAX_STEP, d), arg2: (1, )
            arg1, arg2 = args
            return tf.gather(arg1, arg2, axis=0)

        # (N2, 1)
        ind_sel = tf.reshape(track_len - 1, [-1, 1])
        c_app_states = tf.map_fn(map_op, (c_app_states_all, ind_sel), dtype=tf.float32)
        
        return c_app_states


class app_gating_net_BLSTM(network):

    def __init__(self, FLAGS, cfg, mode):

        super(app_gating_net_BLSTM, self).__init__(FLAGS, cfg, mode)
        self.plh = {}
        self.var = {}
        self.setup_plh()
        self.setup_var()
        
    def setup_plh(self):

        images_shape = [None, self.cfg.IMAGE_HEIGHT, self.cfg.IMAGE_WIDTH, 3]
        self.plh['images'] = tf.placeholder(tf.float32, shape=images_shape)
        self.plh['is_training'] = tf.placeholder(tf.bool, [], name='is_training')
        self.plh['num_step_by_user'] = tf.placeholder(tf.int32, [], name='num_step')
        self.plh['frames_by_user'] = tf.placeholder(tf.int32, [1], name='frame_sel')
        # indices for building memory that contains negative target information
        self.plh['indices_by_user'] = tf.placeholder(tf.int32, [None, None, None, 2], name='frame_sel')
        self.plh['track_len'] = tf.placeholder(tf.int32, [None], name='track_len')
        self.plh['track_len_offset'] = tf.placeholder(tf.int32, [None], name='track_len_offset')
        self.plh['sel_indices'] = tf.placeholder(tf.float32, [None, None, None], name='select_data')
        self.plh['valid_app_data'] = tf.placeholder(tf.float32, [None, None, 1], name='valid_app_data')
        self.plh['indices_for_mapping'] = tf.placeholder(tf.int32, [None, 1], name='mapping_ind')
        self.plh['image_batch_shape'] = tf.placeholder(tf.int32, [None], name='image_batch_shape')
        self.plh['sel_pool_indices'] = tf.placeholder(tf.float32, [None, None, 1])
        # memories for tracks that started earlier than the current frame
        self.plh['prev_track_memory'] = tf.placeholder(tf.float32, [None, self.cfg.APP_HIDDEN_DIM], 'prev_mem')
        self.plh['prev_track_ind'] = tf.placeholder(tf.int32, [None], name='prev_mem_ind')
        self.plh['nonprev_track_ind'] = tf.placeholder(tf.int32, [None], name='nonprev_mem_ind')
        # container for concated lstm states
        self.plh['istate_app'] = tf.placeholder(tf.float32, [None, 2 * self.cfg.APP_HIDDEN_DIM])
        # needed in test mode
        self.plh['app_embed_plh'] = tf.placeholder(tf.float32, shape=[None, 1, self.cfg.APP_LAYER_DIM])
        self.plh['h_app_track_memory_plh'] = tf.placeholder(tf.float32, [None, 1, self.cfg.APP_HIDDEN_DIM])
        self.plh['orig_noisy_bboxes_synced_with_app'] = tf.placeholder(tf.float32, shape=[None, self.cfg.MOT_INPUT_DIM])
        self.plh['app_frame_num'] = tf.placeholder(tf.int32, shape=[None, None])
        self.plh['trk_bbox_org_prev'] = tf.placeholder(tf.float32, shape=[None, self.cfg.MOT_INPUT_DIM])

        if self.mode == 'test':
            self.plh['detbb_num_test'] = tf.placeholder(tf.int32, [])
            self.plh['track_num_test'] = tf.placeholder(tf.int32, [])
            self.plh['trk_bbox_org'] = tf.placeholder(tf.float32, shape=[None, self.cfg.MOT_INPUT_DIM])
            self.plh['det_bbox_org'] = tf.placeholder(tf.float32, shape=[None, self.cfg.MOT_INPUT_DIM])

    def setup_var(self):

        if self.mode == 'train':
            self.var['detbb_num'] = tf.shape(self.plh['images'])[0] / self.plh['num_step_by_user']
            self.var['track_num'] = tf.shape(self.plh['istate_app'])[0]
            self.var['num_step_check'] = tf.shape(self.plh['images'])[0] / self.var['detbb_num']
        else:
            self.var['detbb_num'] = self.plh['detbb_num_test']
            self.var['track_num'] = self.plh['track_num_test']

    def predict(self):

        self.var['app_embed'] = self.build_cnn()

        if self.mode == 'train':
            app_embed = self.var['app_embed']
            self.var['h_app_track_memory'] = self.build_lstm_track_memory(app_embed)
            self.var['h_app_track_memory_plh'] = self.var['h_app_track_memory']
        else:
            app_embed = self.plh['app_embed_plh']
            self.var['h_app_track_memory'] = self.build_lstm_track_memory(app_embed)
            self.var['h_app_track_memory_plh'] = self.plh['h_app_track_memory_plh']
        
        predictions, _ = self.build_blstm_predictor(app_embed)

        return predictions

    def collect_trainer_ops(self, losses, train_op, logging_ops):

        bookkeeping = self.bookkeeping

        total_loss, reg_loss, cls_loss = losses
        savers = logging_ops['savers']
        train_log_ops = logging_ops['train_log_ops']
        val_log_ops = logging_ops['val_log_ops']
        
        tf_ops = {}
        tf_ops['train_op'] = train_op
        tf_ops['cls_loss'] = cls_loss
        tf_ops['reg_loss'] = reg_loss
        tf_ops['total_loss'] = total_loss
        tf_ops['accuracy_total'] = bookkeeping['accuracy_total']
        tf_ops['labels_vec'] = bookkeeping['labels_vec']
        tf_ops['predicted_classes'] = bookkeeping['predicted_classes']
        tf_ops['detbb_num'] = self.var['detbb_num'] 
        tf_ops['num_step_check'] = self.var['num_step_check'] 
        tf_ops['frames_by_user'] = self.plh['frames_by_user']
        tf_ops['h_app_states_first_sel'] = bookkeeping['h_app_states_first_sel']
        tf_ops['c_app_states_first_sel'] = bookkeeping['c_app_states_first_sel']
        tf_ops['c_app_states_last_sel'] = bookkeeping['c_app_states_last_sel']
        tf_ops['c_app_states'] = bookkeeping['c_app_states']
        # tf_ops['bb_test'] = bookkeeping['bb_test']
        # tf_ops['bb_test2'] = bookkeeping['bb_test2']
        tf_ops['app_trk_bbox_frame'] = bookkeeping['app_trk_bbox_frame_for_verification']
        tf_ops['app_det_bbox_frame'] = bookkeeping['app_det_bbox_frame_for_verification']
        tf_ops['app_lstm_state_frame'] = bookkeeping['app_lstm_state_frame_for_verification']
        tf_ops['app_det_bbox_associated_with_init_state'] = bookkeeping['app_det_bbox_associated_with_init_state']

        train_log_ops['h_app_states_first_sel'] = bookkeeping['h_app_states_first_sel']
        train_log_ops['c_app_states_first_sel'] = bookkeeping['c_app_states_first_sel']
        train_log_ops['app_lstm_state_frame'] = bookkeeping['app_lstm_state_frame_for_verification']
        train_log_ops['app_det_bbox_associated_with_init_state'] = bookkeeping['app_det_bbox_associated_with_init_state']
        val_log_ops['h_app_states_first_sel'] = bookkeeping['h_app_states_first_sel']
        val_log_ops['c_app_states_first_sel'] = bookkeeping['c_app_states_first_sel']
        val_log_ops['app_lstm_state_frame'] = bookkeeping['app_lstm_state_frame_for_verification']
        val_log_ops['app_det_bbox_associated_with_init_state'] = bookkeeping['app_det_bbox_associated_with_init_state']

        tf_placeholders = {}
        tf_placeholders['images'] = self.plh['images']
        tf_placeholders['istate_app'] = self.plh['istate_app']
        tf_placeholders['is_training'] = self.plh['is_training']
        tf_placeholders['num_step_by_user'] = self.plh['num_step_by_user']
        tf_placeholders['frames_by_user'] = self.plh['frames_by_user']
        tf_placeholders['indices_by_user'] = self.plh['indices_by_user']
        tf_placeholders['track_len'] = self.plh['track_len']
        tf_placeholders['sel_indices'] = self.plh['sel_indices']
        tf_placeholders['valid_app_data'] = self.plh['valid_app_data']
        tf_placeholders['indices_for_mapping'] = self.plh['indices_for_mapping']
        tf_placeholders['image_batch_shape'] = self.plh['image_batch_shape']
        tf_placeholders['track_len_offset'] = self.plh['track_len_offset']
        tf_placeholders['sel_pool_indices'] = self.plh['sel_pool_indices']
        tf_placeholders['prev_track_memory'] = self.plh['prev_track_memory']
        tf_placeholders['prev_track_ind'] = self.plh['prev_track_ind']
        tf_placeholders['nonprev_track_ind'] = self.plh['nonprev_track_ind']
        tf_placeholders['orig_noisy_bboxes_synced_with_app'] = self.plh['orig_noisy_bboxes_synced_with_app']
        tf_placeholders['app_frame_num'] = self.plh['app_frame_num']
        tf_placeholders['trk_bbox_org_prev']  = self.plh['trk_bbox_org_prev']

        tf_vars = [tf_ops, train_log_ops, val_log_ops, tf_placeholders]

        return (tf_vars, savers)

    def collect_tracker_ops(self, cls_prediction_vec):
        
        # (N2 * N1 * T) x output_dim
        softmax_out = tf.nn.softmax(cls_prediction_vec)

        tf_logging_ops = self.get_logging_op()
        savers = tf_logging_ops['savers']

        tf_ops = {}
        tf_ops['softmax_out'] = softmax_out
        tf_ops['h_app_states'] = self.bookkeeping['h_app_states']
        tf_ops['c_app_states'] = self.bookkeeping['c_app_states']
        tf_ops['h_app_track_memory'] = self.var['h_app_track_memory']
        tf_ops['m_states'] = self.bookkeeping['m_states']
        tf_ops['app_embed'] = self.var['app_embed']

        tf_placeholders = {}
        tf_placeholders['images'] = self.plh['images']
        tf_placeholders['istate_app'] = self.plh['istate_app']
        tf_placeholders['is_training'] = self.plh['is_training']
        tf_placeholders['num_step_by_user'] = self.plh['num_step_by_user']
        tf_placeholders['frames_by_user'] = self.plh['frames_by_user']
        tf_placeholders['indices_by_user'] = self.plh['indices_by_user']
        tf_placeholders['sel_pool_indices'] = self.plh['sel_pool_indices']
        tf_placeholders['h_app_track_memory_plh'] = self.plh['h_app_track_memory_plh']
        tf_placeholders['track_len'] = self.plh['track_len']
        tf_placeholders['app_embed_plh'] = self.plh['app_embed_plh']
        tf_placeholders['valid_app_data'] = self.plh['valid_app_data']
        tf_placeholders['indices_for_mapping'] = self.plh['indices_for_mapping']
        tf_placeholders['image_batch_shape'] = self.plh['image_batch_shape']
        tf_placeholders['track_len_offset'] = self.plh['track_len_offset']
        tf_placeholders['det_bbox_org'] = self.plh['det_bbox_org']
        tf_placeholders['trk_bbox_org'] = self.plh['trk_bbox_org']
        tf_placeholders['app_frame_num'] = self.plh['app_frame_num']
        tf_placeholders['sel_indices'] = self.plh['sel_indices']
        tf_placeholders['detbb_num'] = self.plh['detbb_num_test']
        tf_placeholders['track_num'] = self.plh['track_num_test']

        tf_vars = [tf_ops, tf_placeholders]

        return (tf_vars, savers)

    def build_cnn(self):

        valid_mask = tf.cast(tf.reshape(self.plh['valid_app_data'], [-1]), tf.bool)
        images = tf.boolean_mask(self.plh['images'], valid_mask)
        original_ind = tf.boolean_mask(self.plh['indices_for_mapping'], valid_mask)

        # build the network
        with slim.arg_scope(resnet_v1.resnet_arg_scope(
                                batch_norm_decay=self.FLAGS.batch_norm_decay, 
                                weight_decay=self.FLAGS.weight_decay)
        ):
            cnn_features, end_points = resnet_v1.resnet_v1_50(
                                           images,
                                           is_training=self.plh['is_training'],
                                           output_stride=None)
            features = tf.squeeze(cnn_features, [1, 2], name='SpatialSqueeze')

        # reduce the feat dimension
        with tf.variable_scope('app_lstm/app_embedding'):
            weights_app = self.init_weights('weights', self.FLAGS.weight_decay, [2048, self.cfg.APP_LAYER_DIM])
            biases_app = self.init_biases('biases', [self.cfg.APP_LAYER_DIM])
            app_embed = tf.nn.relu(tf.matmul(features, weights_app) + biases_app)

            # zero padding 
            app_embed = tf.scatter_nd(original_ind, app_embed, self.plh['image_batch_shape'])

            # N x sequence length x feature dimension
            app_embed = tf.reshape(app_embed, [self.var['detbb_num'], -1, self.cfg.APP_LAYER_DIM])

        return app_embed

    def build_lstm_track_memory(self, app_embed):

        # create track memory vectors using LSTM
        with tf.variable_scope('app_lstm/lstm'):
            # num_step = tf.reshape(num_step_scalar, [1,])
            # repeat = tf.reshape(batch_num, [1,])
            # seq_len = tf.tile(num_step, repeat)

            num_step_pad = self.cfg.MAX_STEP - self.plh['num_step_by_user']
            pad_sz = [self.var['track_num'], num_step_pad, self.cfg.APP_LAYER_DIM]
            zero_pad_app = tf.zeros(pad_sz)

            data_holder_sz = [self.var['track_num'], self.cfg.MAX_STEP, self.cfg.APP_LAYER_DIM]
            data_app_padded = tf.concat([app_embed, zero_pad_app], 1)
            data_app_padded = tf.reshape(data_app_padded, data_holder_sz)

            lstm_init = tf.truncated_normal_initializer(mean=0.0, stddev=0.02)
            lstm_cell_tmp = LSTMCell(
                                self.cfg.APP_HIDDEN_DIM,
                                initializer=lstm_init,
                                state_is_tuple=False
                            )
            lstm_cell = Wrapper(lstm_cell_tmp)

            # N2 x MAX_STEP x d (N2: # of tracks, d: lstm state dimension)
            h_app_states_pre, c_app_states = tf.nn.dynamic_rnn(
                                                lstm_cell,
                                                data_app_padded,
                                                initial_state=self.plh['istate_app'],
                                                sequence_length=self.plh['track_len']
                                            )

            # if I use the custom wrapper
            c_app_states_all, h_app_states_pre = h_app_states_pre

            if self.mode == 'train':
                c_app_states_first_sel = c_app_states_all[:, 0, :]
                c_app_states_last_sel_tmp = self.gather_final_states(c_app_states_all, self.plh['track_len'])
                c_app_states_last_sel = tf.reshape(c_app_states_last_sel_tmp, [self.var['track_num'], -1])
                h_app_states_first_sel = h_app_states_pre[:, 0, :]

                # Let's align data in axis 1
                # Example:
                # (Before alignment)
                # ---------- track 1 ------------
                # ----- track 2 ----- 00000000000
                # -------- track 3 ------- 000000
                # (After alignment)
                # ---------- track 1 ------------
                # 00000000000 ----- track 2 -----
                # 000000 -------- track 3 -------
                h_app_states_aft = self.align_with_frames(
                                    h_app_states_pre, 
                                    self.plh['track_len'] + self.plh['track_len_offset'],
                                    self.plh['num_step_by_user']
                                )
            else:
                c_app_states_first_sel = c_app_states_all[:, 0, :]
                c_app_states_last_sel = c_app_states_all[:, 0, :]
                h_app_states_first_sel = h_app_states_pre[:, 0, :]
                h_app_states_aft = h_app_states_pre

            app_frame_num = tf.reshape(self.plh['app_frame_num'], [self.var['track_num'], self.plh['num_step_by_user'], 1])
            app_frame_num = tf.cast(app_frame_num, tf.float32)
            app_lstm_state_frame_for_verification = app_frame_num[:, 0, :]

            app_frame_num_aligned = self.align_with_frames(
                                        app_frame_num,
                                        self.plh['track_len'] + self.plh['track_len_offset'],
                                        self.plh['num_step_by_user']
                                    )

            detection_bbox_org = tf.reshape(self.plh['orig_noisy_bboxes_synced_with_app'], [self.var['track_num'], self.plh['num_step_by_user'], self.cfg.MOT_INPUT_DIM])
            detection_bbox_associated_with_init_state = detection_bbox_org[:, 0, :]

            # N2 x T x d (T: # of frames selected for training)
            prev_fames_by_user = self.plh['frames_by_user'] - 1
            h_app_states = tf.gather(h_app_states_aft, prev_fames_by_user, axis=1)

            app_trk_bbox_frame_for_verification = tf.gather(app_frame_num_aligned, prev_fames_by_user, axis=1)
            app_trk_bbox_frame_for_verification = tf.reshape(app_trk_bbox_frame_for_verification, [self.var['track_num']])

            if self.mode == 'train':
                with tf.control_dependencies(
                    [tf.assert_equal(tf.reduce_sum(tf.gather(h_app_states, self.plh['prev_track_ind'], axis=0)), 0.0),
                    tf.assert_equal(tf.reduce_sum(tf.gather(self.plh['prev_track_memory'], self.plh['nonprev_track_ind'], axis=0)), 0.0)]):          
                    prev_track_memory = tf.reshape(self.plh['prev_track_memory'], [-1, 1, self.cfg.APP_HIDDEN_DIM])
                    h_app_track_memory = h_app_states + prev_track_memory
            elif self.mode == 'test':
                h_app_track_memory = h_app_states
            else:
                raise NotImplementedError
    
        self.bookkeeping['h_app_states_first_sel'] = h_app_states_first_sel
        self.bookkeeping['h_app_states_aft'] = h_app_states_aft
        self.bookkeeping['h_app_states_pre'] = h_app_states_pre
        self.bookkeeping['h_app_states'] = h_app_states
        self.bookkeeping['c_app_states'] = c_app_states
        self.bookkeeping['track_len'] = self.plh['track_len']
        self.bookkeeping['c_app_states_first_sel'] = c_app_states_first_sel
        self.bookkeeping['c_app_states_last_sel'] = c_app_states_last_sel
        self.bookkeeping['app_trk_bbox_frame_for_verification'] = app_trk_bbox_frame_for_verification
        self.bookkeeping['app_lstm_state_frame_for_verification'] = app_lstm_state_frame_for_verification
        self.bookkeeping['app_det_bbox_associated_with_init_state'] = detection_bbox_associated_with_init_state
        
        return h_app_track_memory

    def build_blstm_predictor(self, app_embed_plh):

        # final predictions
        with tf.variable_scope("app_lstm/output"):

            if self.mode == 'train':
                app_embed = self.align_with_frames(
                                app_embed_plh,
                                self.plh['track_len'] + self.plh['track_len_offset'],
                                self.plh['num_step_by_user']
                            )
            elif self.mode == 'test':
                app_embed = app_embed_plh
            else:
                raise NotImplementedError

            app_frame_num = tf.reshape(self.plh['app_frame_num'], [self.var['track_num'], self.plh['num_step_by_user'], 1])
            app_frame_num = tf.cast(app_frame_num, tf.float32)
            app_frame_num_aligned = self.align_with_frames(
                                        app_frame_num,
                                        self.plh['track_len'] + self.plh['track_len_offset'],
                                        self.plh['num_step_by_user']
                                    )

            # N1 x T x feat_dim (N1: # of detections)
            app_embed = tf.gather(app_embed, self.plh['frames_by_user'], axis=1)
            
            app_det_bbox_frame_for_verification = tf.gather(app_frame_num_aligned, self.plh['frames_by_user'], axis=1)
            app_det_bbox_frame_for_verification = tf.reshape(app_det_bbox_frame_for_verification, [self.var['detbb_num']])

            # N1 x T x 1 x feat_dim
            app_embed = tf.reshape(
                            app_embed,
                            [self.var['detbb_num'], tf.shape(self.var['h_app_track_memory_plh'])[1], 1, self.cfg.APP_LAYER_DIM])

            blstm_dim = int(self.cfg.APP_HIDDEN_DIM / self.cfg.APP_LAYER_DIM)
            # N2 x T x feat_dim x (d / feat_dim)
            mem_sz = [self.var['track_num'], tf.shape(self.var['h_app_track_memory_plh'])[1], \
                        self.cfg.APP_LAYER_DIM, blstm_dim]
            h_app_track_memory = tf.reshape(self.var['h_app_track_memory_plh'], mem_sz)

            #  1 x N1 x T x 1 x feat_dim
            app_embed = tf.expand_dims(app_embed, axis=0)
            # N2 x N1 x T x 1 x feat_dim
            app_embed_all = tf.tile(app_embed, [self.var['track_num'], 1, 1, 1, 1])

            # N2 x  1 x T x feat_dim x (d / feat_dim)
            h_app_track_memory = tf.expand_dims(h_app_track_memory, axis=1)
            # N2 x N1 x T x feat_dim x (d / feat_dim)
            h_app_track_memory = tf.tile(h_app_track_memory, [1, self.var['detbb_num'], 1, 1, 1])

            # N2 x N1 x T x 1 x (d / feat_dim) matrix 
            m_states = tf.nn.relu(tf.matmul(app_embed_all, h_app_track_memory))

            # (N2 * N1 * T) x (d / feat_dim) matrix
            m_states = tf.reshape(m_states, [-1, blstm_dim])

            # final weights
            weights_cls = self.init_weights('weights', self.FLAGS.weight_decay,
                                            [blstm_dim, self.cfg.OUT_NUM])
            biases_cls = self.init_biases('biases', [self.cfg.OUT_NUM])

            # (N2 * N1 * T) x output_dim
            cls_prediction_vec = tf.matmul(m_states, weights_cls) + biases_cls

            # N2 x N1 x T x output_dim
            cls_prediction = tf.reshape(
                                 cls_prediction_vec,
                                 [self.var['track_num'], self.var['detbb_num'], -1, self.cfg.OUT_NUM])
        
            predictions = cls_prediction, cls_prediction_vec

        self.bookkeeping['app_embed_all'] = app_embed_all
        self.bookkeeping['m_states'] = m_states
        self.bookkeeping['cls_prediction_vec'] = cls_prediction_vec
        self.bookkeeping['app_det_bbox_frame_for_verification'] = app_det_bbox_frame_for_verification

        return (predictions, m_states)


class app_gating_net_BLSTM_MTP(app_gating_net_BLSTM):

    def __init__(self, FLAGS, cfg, mode):

        super(app_gating_net_BLSTM_MTP, self).__init__(FLAGS, cfg, mode)

    def predict(self):

        self.var['app_embed'] = self.build_cnn()

        if self.mode == 'train':
            app_embed = self.var['app_embed']
            self.var['h_app_track_memory'] = self.build_lstm_track_memory(app_embed)
            self.var['h_app_track_memory_plh'] = self.var['h_app_track_memory']
        else:
            app_embed = self.plh['app_embed_plh']
            self.var['h_app_track_memory'] = self.build_lstm_track_memory(app_embed)
            self.var['h_app_track_memory_plh'] = self.plh['h_app_track_memory_plh']

        predictions, _ = self.build_blstm_mtp_predictor(app_embed)

        return predictions

    def build_blstm_mtp_predictor(self, app_embed_plh):

        # final predictions
        with tf.variable_scope("app_lstm/output"):

            if self.mode == 'train':
                app_embed = self.align_with_frames(
                                app_embed_plh,
                                self.plh['track_len'] + self.plh['track_len_offset'],
                                self.plh['num_step_by_user']
                            )
            elif self.mode == 'test':
                app_embed = app_embed_plh
            else:
                raise NotImplementedError

            app_frame_num = tf.reshape(self.plh['app_frame_num'], [self.var['track_num'], self.plh['num_step_by_user'], 1])
            app_frame_num = tf.cast(app_frame_num, tf.float32)
            app_frame_num_aligned = self.align_with_frames(
                                        app_frame_num,
                                        self.plh['track_len'] + self.plh['track_len_offset'],
                                        self.plh['num_step_by_user']
                                    )

            # N1 x T x feat_dim (N1: # of detections)
            app_embed = tf.gather(app_embed, self.plh['frames_by_user'], axis=1)
            
            app_det_bbox_frame_for_verification = tf.gather(app_frame_num_aligned, self.plh['frames_by_user'], axis=1)
            app_det_bbox_frame_for_verification = tf.reshape(app_det_bbox_frame_for_verification, [self.var['detbb_num']])

            # N1 x T x 1 x feat_dim
            app_embed = tf.reshape(
                            app_embed, 
                            [self.var['detbb_num'], tf.shape(self.var['h_app_track_memory_plh'])[1], 1, self.cfg.APP_LAYER_DIM])

            blstm_dim = int(self.cfg.APP_HIDDEN_DIM / self.cfg.APP_LAYER_DIM)
            # N2 x T x feat_dim x (d / feat_dim)
            mem_sz = [self.var['track_num'], tf.shape(self.var['h_app_track_memory_plh'])[1], \
                        self.cfg.APP_LAYER_DIM, blstm_dim]
            h_app_track_memory = tf.reshape(self.var['h_app_track_memory_plh'], mem_sz)

            #  1 x N1 x T x 1 x feat_dim
            app_embed = tf.expand_dims(app_embed, axis=0)
            # N2 x N1 x T x 1 x feat_dim
            app_embed_all = tf.tile(app_embed, [self.var['track_num'], 1, 1, 1, 1])

            # N2 x  1 x T x feat_dim x (d / feat_dim)
            h_app_track_memory = tf.expand_dims(h_app_track_memory, axis=1)
            # N2 x N1 x T x feat_dim x (d / feat_dim)
            h_app_track_memory = tf.tile(h_app_track_memory, [1, self.var['detbb_num'], 1, 1, 1])

            # N2 x N1 x T x 1 x (d / feat_dim) matrix 
            m_states = tf.nn.relu(tf.matmul(app_embed_all, h_app_track_memory))

            # N2 x N1 x (d / feat_dim) matrix (T = 1)
            m_states = tf.reshape(m_states, [self.var['track_num'], self.var['detbb_num'], blstm_dim])

            # building negative memory - same detection, different (N2 - 1) tracks                
            # N2 x N1 x (N2 - 1) x (d / feat_dim) matrix
            neg_raw_mem = tf.gather_nd(m_states, self.plh['indices_by_user'])
            # (N2 * N1) x (d / feat_dim) matrix
            m_states = tf.reshape(m_states, [self.var['track_num'] * self.var['detbb_num'], blstm_dim])
            # (N2 * N1) x (N2 - 1) x (d / feat_dim) matrix
            neg_raw_mem = tf.reshape(neg_raw_mem, [self.var['track_num'] * self.var['detbb_num'], self.var['track_num'] - 1, blstm_dim])
            
            if self.mode == 'train':
                augmentation_mask = self.generate_augmentation_mask(self.plh['indices_by_user'])
                neg_raw_mem = tf.multiply(neg_raw_mem, augmentation_mask)
  
            # (N2 * N1 * T) x (d / feat_dim) matrix
            neg_mem = tf.reduce_max(neg_raw_mem, axis=1)

            # concatenate positive and negative memory units
            # (N2 * N1 * T) x (2 * (d / feat_dim)) matrix
            m_states = tf.concat([m_states, neg_mem], axis=1)

            # final weights
            weights_cls = self.init_weights('weights', self.FLAGS.weight_decay,
                                            [2 * blstm_dim, self.cfg.OUT_NUM])
            biases_cls = self.init_biases('biases', [self.cfg.OUT_NUM])

            # (N2 * N1 * T) x output_dim
            cls_prediction_vec = tf.matmul(m_states, weights_cls) + biases_cls

            # N2 x N1 x T x output_dim
            cls_prediction = tf.reshape(
                                 cls_prediction_vec,
                                 [self.var['track_num'], self.var['detbb_num'], -1, self.cfg.OUT_NUM])
        
            predictions = cls_prediction, cls_prediction_vec

        self.bookkeeping['app_embed_all'] = app_embed_all
        self.bookkeeping['neg_raw_mem'] = neg_raw_mem
        self.bookkeeping['neg_mem'] = neg_mem
        self.bookkeeping['m_states'] = m_states
        self.bookkeeping['cls_prediction_vec'] = cls_prediction_vec
        self.bookkeeping['app_det_bbox_frame_for_verification'] = app_det_bbox_frame_for_verification

        return (predictions, m_states)

    def generate_augmentation_mask(self, ind_sel):

        # create labels
        idx = tf.range(self.var['detbb_num'])
        idx = tf.expand_dims(idx, axis=0)
        # N2 x N1
        idx = tf.tile(idx, [self.var['track_num'], 1])
        # N2 x N1 x 1
        idx = tf.expand_dims(idx, axis=2)
        # N2 x N1 x (N2 - 1)
        idx = tf.tile(idx, [1, 1, self.var['track_num'] - 1])

        # N2 x N1 x (N2 - 1)
        augmentation_mask = tf.not_equal(ind_sel[:, :, :, 0], idx)
        augmentation_mask = tf.cast(augmentation_mask, dtype=tf.int32)

        select = tf.random.uniform(shape=[self.var['detbb_num'], 1], minval=0., maxval=1., dtype=tf.float32)
        # N2 x N1 x 1
        select = tf.cast(select > self.cfg.NEW_TRACK_AUG_RATE, dtype=tf.int32)
        select = tf.expand_dims(select, axis=0)
        select = tf.tile(select, [self.var['track_num'], 1, 1])
        augmentation_mask = tf.add(augmentation_mask, select)
        augmentation_mask = tf.cast(augmentation_mask, dtype=tf.bool)
        augmentation_mask = tf.cast(augmentation_mask, dtype=tf.float32)
        # (N2 x N1) x (N2 - 1) x 1
        augmentation_mask = tf.reshape(augmentation_mask, [self.var['track_num']* self.var['detbb_num'], self.var['track_num'] - 1, 1])

        return augmentation_mask


class mot_gating_net_LSTM(network):

    def __init__(self, FLAGS, cfg, mode):

        super(mot_gating_net_LSTM, self).__init__(FLAGS, cfg, mode)
        self.plh = {}
        self.var = {}
        self.setup_plh()
        self.setup_var()

    def setup_plh(self):

        self.plh['detection_bboxes'] = tf.placeholder(tf.float32, shape=(None, self.cfg.MOT_INPUT_DIM))       
        self.plh['valid_mot_data'] = tf.placeholder(tf.float32, shape=[None, None, 1])
        self.plh['track_len'] = tf.placeholder(tf.int32, shape=[None])
        self.plh['start_offset'] = tf.placeholder(tf.int32, shape=[None])
        self.plh['end_offset'] = tf.placeholder(tf.int32, shape=[None])
        self.plh['mid_missdet_num'] = tf.placeholder(tf.int32, shape=[None])
        self.plh['first_missdet_num'] = tf.placeholder(tf.int32, shape=[None])
        self.plh['num_step_by_user'] = tf.placeholder(tf.int32, [], name='num_step')
        self.plh['frames_by_user'] = tf.placeholder(tf.int32, [1], name='frame_sel')
        self.plh['sel_indices'] = tf.placeholder(tf.float32, [None, None, None], name='select_data')
        self.plh['istate_mot'] = tf.placeholder(tf.float32, [None, 2 * self.cfg.MOT_HIDDEN_DIM])
        self.plh['c_mot_states_plh'] = tf.placeholder(tf.float32, [None, 2 * self.cfg.MOT_HIDDEN_DIM])
        self.plh['orig_noisy_bboxes_synced_with_mot'] = tf.placeholder(tf.float32, shape=[None, self.cfg.MOT_INPUT_DIM])
        self.plh['orig_noisy_bboxes_prev_synced_with_mot'] = tf.placeholder(tf.float32, shape=[None, self.cfg.MOT_INPUT_DIM])
        self.plh['mot_frame_num'] = tf.placeholder(tf.int32, shape=[None, None])
        self.plh['trk_bbox_org_prev'] = tf.placeholder(tf.float32, shape=[None, self.cfg.MOT_INPUT_DIM])

        if self.mode == 'test':
            self.plh['detbb_num_test'] = tf.placeholder(tf.int32, [])
            self.plh['track_num_test'] = tf.placeholder(tf.int32, [])
            self.plh['trk_bbox_org'] = tf.placeholder(tf.float32, shape=[None, self.cfg.MOT_INPUT_DIM])
            self.plh['det_bbox_org'] = tf.placeholder(tf.float32, shape=[None, self.cfg.MOT_INPUT_DIM])

    def setup_var(self):
    
        if self.mode == 'train':
            self.var['detbb_num'] = tf.shape(self.plh['detection_bboxes'])[0] / self.plh['num_step_by_user']
            self.var['track_num'] = tf.shape(self.plh['istate_mot'])[0]        
            # currently, I assume that detbb_num == track_num
            with tf.control_dependencies([tf.assert_equal(self.var['detbb_num'], self.var['track_num'])]):
                track_num = tf.shape(self.plh['istate_mot'])[0]
        elif self.mode == 'test':
            self.var['detbb_num'] = self.plh['detbb_num_test']
            self.var['track_num'] = self.plh['track_num_test']
        
        self.var['num_step'] = tf.shape(tf.reshape(self.plh['detection_bboxes'], 
                                           [self.var['detbb_num'], -1, self.cfg.MOT_INPUT_DIM]))[1]
        self.var['num_step_scalar'] = tf.reshape(self.var['num_step'], [1,])

    def predict(self):

        bbox_embed = self.compute_mot_features()
        bbox_embed_wo_last, bbox_embed_last = self.generate_matching_pairs(bbox_embed)
        hidden_states_mot = self.build_mot_lstm_memory(bbox_embed_wo_last, bbox_embed_last)
        predictions, _ = self.build_mot_lstm_predictor(hidden_states_mot)
        
        return predictions

    def collect_trainer_ops(self, losses, train_op, logging_ops):

        bookkeeping = self.bookkeeping

        total_loss, reg_loss, cls_loss = losses
        savers = logging_ops['savers']
        train_log_ops = logging_ops['train_log_ops']
        val_log_ops = logging_ops['val_log_ops']

        tf_ops = {}
        tf_ops['train_op'] = train_op
        tf_ops['cls_loss'] = cls_loss
        tf_ops['reg_loss'] = reg_loss
        tf_ops['total_loss'] = total_loss
        tf_ops['accuracy_total'] = bookkeeping['accuracy_total']
        tf_ops['labels_vec'] = bookkeeping['labels_vec']
        tf_ops['predicted_classes'] = bookkeeping['predicted_classes']
        tf_ops['detbb_num'] = self.var['detbb_num']
        tf_ops['num_step'] = self.var['num_step']
        tf_ops['frames_by_user'] = self.plh['frames_by_user']
        tf_ops['c_mot_states_first_sel'] = bookkeeping['c_mot_states_first_sel']
        tf_ops['c_mot_states_last_sel'] = bookkeeping['c_mot_states_last_sel']
        tf_ops['c_mot_states'] = bookkeeping['c_mot_states']
        tf_ops['seq_len_check_old'] = bookkeeping['seq_len_check_old']
        tf_ops['bbox_embed_wo_last_aligned'] = bookkeeping['bbox_embed_wo_last_aligned']
        tf_ops['mid_missdet_num'] = bookkeeping['mid_missdet_num']
        tf_ops['end_offset'] = bookkeeping['e_offset']
        tf_ops['track_len'] = bookkeeping['track_len']
        tf_ops['c_mot_states_all'] = bookkeeping['c_mot_states_all']
        tf_ops['mot_trk_bbox_frame'] = bookkeeping['mot_trk_bbox_frame_for_verification']
        tf_ops['mot_det_bbox_frame'] = bookkeeping['mot_det_bbox_frame_for_verification']
        tf_ops['mot_det_bbox_frame_minus_one'] = bookkeeping['mot_det_bbox_frame_minus_one']
        tf_ops['mot_lstm_state_frame'] = bookkeeping['mot_lstm_state_frame_for_verification']
        tf_ops['mot_det_bbox_associated_with_init_state'] = bookkeeping['mot_det_bbox_associated_with_init_state']
        tf_ops['bbox_embed_masked'] = self.bookkeeping['bbox_embed_masked']

        train_log_ops['c_mot_states_first_sel'] = bookkeeping['c_mot_states_first_sel']
        train_log_ops['neg_start_offset'] = bookkeeping['neg_start_offset']
        train_log_ops['mot_lstm_state_frame'] = bookkeeping['mot_lstm_state_frame_for_verification']
        train_log_ops['mot_det_bbox_associated_with_init_state'] = bookkeeping['mot_det_bbox_associated_with_init_state']
        val_log_ops['c_mot_states_first_sel'] = bookkeeping['c_mot_states_first_sel']
        val_log_ops['mot_lstm_state_frame'] = bookkeeping['mot_lstm_state_frame_for_verification']
        val_log_ops['mot_det_bbox_associated_with_init_state'] = bookkeeping['mot_det_bbox_associated_with_init_state']

        tf_placeholders = {}
        tf_placeholders['detection_bboxes'] = self.plh['detection_bboxes']
        tf_placeholders['valid_mot_data'] = self.plh['valid_mot_data']
        tf_placeholders['frames_by_user'] = self.plh['frames_by_user']
        tf_placeholders['start_offset'] = self.plh['start_offset']
        tf_placeholders['end_offset'] = self.plh['end_offset']
        tf_placeholders['num_step_by_user'] = self.plh['num_step_by_user']
        tf_placeholders['sel_indices'] = self.plh['sel_indices']
        tf_placeholders['istate_mot'] = self.plh['istate_mot']
        tf_placeholders['track_len'] = self.plh['track_len']
        tf_placeholders['mid_missdet_num'] = self.plh['mid_missdet_num']
        tf_placeholders['first_missdet_num'] = self.plh['first_missdet_num']
        tf_placeholders['orig_noisy_bboxes_synced_with_mot'] = self.plh['orig_noisy_bboxes_synced_with_mot']
        tf_placeholders['orig_noisy_bboxes_prev_synced_with_mot'] = self.plh['orig_noisy_bboxes_prev_synced_with_mot']
        tf_placeholders['mot_frame_num'] = self.plh['mot_frame_num']
        tf_placeholders['trk_bbox_org_prev']  = self.plh['trk_bbox_org_prev']

        tf_vars = [tf_ops, train_log_ops, val_log_ops, tf_placeholders]

        return (tf_vars, savers)

    def collect_tracker_ops(self, cls_prediction_vec):
        
        # (N2 * N1 * T) x output_dim
        softmax_out = tf.nn.softmax(cls_prediction_vec)

        tf_logging_ops = self.get_logging_op()
        savers = tf_logging_ops['savers']

        tf_ops = {}
        tf_ops['softmax_out'] = softmax_out
        tf_ops['c_mot_states_last'] = self.bookkeeping['c_mot_states_last']
        tf_ops['h_mot_states'] = self.bookkeeping['h_mot_states']
        tf_ops['c_mot_states_last_test'] = self.bookkeeping['c_mot_states_last_test']
        tf_ops['h_mot_states_test'] = self.bookkeeping['h_mot_states_test']

        tf_placeholders = {}
        tf_placeholders['detection_bboxes'] = self.plh['detection_bboxes']
        tf_placeholders['valid_mot_data'] = self.plh['valid_mot_data']
        tf_placeholders['frames_by_user'] = self.plh['frames_by_user']
        tf_placeholders['start_offset'] = self.plh['start_offset']
        tf_placeholders['end_offset'] = self.plh['end_offset']
        tf_placeholders['num_step_by_user'] = self.plh['num_step_by_user']
        tf_placeholders['istate_mot'] = self.plh['istate_mot']
        tf_placeholders['track_len'] = self.plh['track_len']
        tf_placeholders['c_mot_states_plh'] = self.plh['c_mot_states_plh']
        tf_placeholders['mid_missdet_num'] = self.plh['mid_missdet_num']
        tf_placeholders['first_missdet_num'] = self.plh['first_missdet_num']
        tf_placeholders['detbb_num'] = self.plh['detbb_num_test']
        tf_placeholders['track_num'] = self.plh['track_num_test']
        tf_placeholders['det_bbox_org'] = self.plh['det_bbox_org']
        tf_placeholders['trk_bbox_org'] = self.plh['trk_bbox_org']
        tf_placeholders['sel_indices'] = self.plh['sel_indices']

        tf_vars = [tf_ops, tf_placeholders]

        return (tf_vars, savers)

    def compute_mot_features(self):
        """
        detection_bboxes: normalized bounding box input 
                          (xmin/img_width, ymin/img_height, width/img_width, height/img_height)
        valid_mot_data: used for setting zero vectors for missing detection
        """
        with tf.variable_scope('mot_lstm/bb_embedding'):
            weights_mot = self.init_weights('weights', self.FLAGS.weight_decay,
                                            [self.cfg.MOT_INPUT_DIM, self.cfg.MOT_LAYER_DIM])
            biases_mot = self.init_biases('biases', [self.cfg.MOT_LAYER_DIM])
            # (N2 x T) x 1
            valid_mot_data = tf.reshape(self.plh['valid_mot_data'], [-1, 1])
            with tf.control_dependencies(
                [tf.assert_equal(tf.shape(valid_mot_data)[0], self.var['detbb_num'] * self.var['num_step'])]):
                # (N2 x T) x mot_feat_dim
                bbox_embed = tf.nn.relu(tf.matmul(self.plh['detection_bboxes'], weights_mot) + biases_mot)
            # generate zero inputs to represent missing detections
            # (N2 x T) x mot_feat_dim
            bbox_embed_masked = tf.multiply(bbox_embed, valid_mot_data)
            # N2  x T x mot_feat_dim
            with tf.control_dependencies(
                [tf.assert_equal(tf.shape(bbox_embed_masked)[0], self.var['detbb_num'] * self.var['num_step'])]):
                bbox_embed_masked = tf.reshape(bbox_embed_masked, 
                                             [self.var['detbb_num'] , -1, self.cfg.MOT_LAYER_DIM])
        
        self.bookkeeping['bbox_embed_masked'] = bbox_embed_masked
        return bbox_embed_masked

    def generate_matching_pairs(self, bbox_embed_masked):

        if self.mode == 'train':
            new_start_offset = self.plh['start_offset'] - self.plh['first_missdet_num']
        
            # align by start offset
            # before alignment
            # ------- track ---------
            # ----- track --------000
            # ----- track -------0000
            # after alignment
            # ------- track ---------
            # 000 --- track ---------
            # 0000 -- track ---------
            bbox_embed_aligned = self.align_by_start_offset(bbox_embed_masked, new_start_offset)
            
            # N2 x (T - 1) x mot_feat_dim
            bbox_embed_wo_last = bbox_embed_aligned[:, :-1, :]

            # N2 x 1 x mot_feat_dim
            bbox_embed_last = bbox_embed_aligned[:, -1, :]
            bbox_embed_last = tf.reshape(bbox_embed_last,
                                    [self.var['detbb_num'], -1, self.cfg.MOT_LAYER_DIM])

            with tf.control_dependencies(
                [tf.assert_equal(tf.shape(bbox_embed_wo_last)[1], self.var['num_step_scalar'] - 1),
                tf.assert_equal(tf.shape(bbox_embed_wo_last)[0], self.var['detbb_num']),
                tf.assert_equal(tf.shape(bbox_embed_last)[1], self.cfg.SAMPLE_NUM_FOR_LOSS),
                tf.assert_equal(tf.shape(bbox_embed_last)[0], self.var['detbb_num'])]
            ):     
                # N2
                neg_start_offset = -new_start_offset
            
            # align by negative start offset
            # N2 x (T - 1) x mot_feat_dim
            bbox_embed_wo_last_aligned = self.align_by_start_offset(bbox_embed_wo_last, neg_start_offset)

            # sync verification
            mot_frame_num = tf.reshape(self.plh['mot_frame_num'], [self.var['track_num'], self.plh['num_step_by_user'], 1])
            mot_frame_num = tf.cast(mot_frame_num, tf.float32)
            mot_frame_num_aligned = self.align_by_start_offset(mot_frame_num, new_start_offset)
            valid_mot_data_aligned = self.align_by_start_offset(self.plh['valid_mot_data'], new_start_offset)
            trk_bbox_frame = mot_frame_num_aligned[:, -2]
            det_bbox_frame = mot_frame_num_aligned[:, -1]
            det_bbox_frame_mask = valid_mot_data_aligned[:, -1]
            det_bbox_frame = tf.multiply(det_bbox_frame, det_bbox_frame_mask)
            trk_bbox_frame = tf.reshape(trk_bbox_frame, [self.var['track_num']])
            det_bbox_frame = tf.reshape(det_bbox_frame, [self.var['detbb_num']])

            self.bookkeeping['bbox_embed_wo_last_aligned'] = bbox_embed_wo_last_aligned
            self.bookkeeping['bbox_embed_last'] = bbox_embed_last
            self.bookkeeping['neg_start_offset'] = neg_start_offset
            self.bookkeeping['mot_trk_bbox_frame_for_verification'] = trk_bbox_frame
            self.bookkeeping['mot_det_bbox_frame_for_verification'] = det_bbox_frame
        else:
            bbox_embed_wo_last_aligned = None
            bbox_embed_last = bbox_embed_masked[:, -1, :]
            bbox_embed_last = tf.reshape(bbox_embed_last, [self.var['detbb_num'], -1, self.cfg.MOT_LAYER_DIM])
            self.bookkeeping['bbox_embed_last'] = bbox_embed_last

        return (bbox_embed_wo_last_aligned, bbox_embed_last)

    def build_mot_lstm_memory(self, bbox_embed_wo_last_aligned, bbox_embed_last):

        with tf.variable_scope('mot_lstm/lstm'):
            lstm_init = tf.truncated_normal_initializer(mean=0.0, stddev=0.02)
            lstm_cell_tmp = LSTMCell(self.cfg.MOT_HIDDEN_DIM,
                                    initializer=lstm_init,
                                    state_is_tuple=False)
            lstm_cell = Wrapper(lstm_cell_tmp)

            if self.mode == 'train':        
                bbox_zero_pad = tf.zeros(
                                  [self.var['track_num'],
                                   (self.cfg.MAX_STEP - 1) - (self.var['num_step'] - 1),
                                   self.cfg.MOT_LAYER_DIM]
                              )
                detection_bboxes_padded = tf.concat([bbox_embed_wo_last_aligned, bbox_zero_pad], 1)
                detection_bboxes_padded = tf.reshape(
                                           detection_bboxes_padded,
                                           [self.var['track_num'], self.cfg.MAX_STEP - 1, self.cfg.MOT_LAYER_DIM])
        
                with tf.control_dependencies(
                    [tf.assert_equal(
                        self.var['num_step_scalar'],
                        self.plh['start_offset'] + self.plh['track_len'] + self.plh['end_offset'] + self.plh['mid_missdet_num'])]
                ):
                    # N2
                    seq_len = (self.plh['track_len'] - 1) + self.plh['end_offset'] + self.plh['mid_missdet_num'] + self.plh['first_missdet_num']  
        
                # for motion network, due to missing detection handling, 
                # "track_len + end_offset" is used to represent track length instead of "track_len" 
                h_mot_states_pre, c_mot_states_old = tf.nn.dynamic_rnn(
                                                        lstm_cell,
                                                        detection_bboxes_padded,
                                                        initial_state=self.plh['istate_mot'],
                                                        sequence_length=seq_len
                                                    )
        
                # parse output from the custom wrapper
                # N2 x (T - 1) x lstm_hidden_dim
                c_mot_states_all, h_mot_states_pre = h_mot_states_pre
                # N2 x (2 x lstm_hidden_dim)
                c_mot_states_first_sel = c_mot_states_all[:, 0, :]
                # N2 x 1 x (2 x lstm_hidden_dim)
                c_mot_states_last_sel_tmp = self.gather_final_states(c_mot_states_all, seq_len)
                assert(self.cfg.SAMPLE_NUM_FOR_LOSS == 1)
                with tf.control_dependencies(
                    [tf.assert_equal(tf.shape(c_mot_states_last_sel_tmp)[1], 1)]):
                    # N2 x (2 x lstm_hidden_dim)
                    c_mot_states_last_sel = tf.reshape(c_mot_states_last_sel_tmp,
                                                       [self.var['track_num'], -1])

            elif self.mode == 'test':
                c_mot_states_old = self.plh['c_mot_states_plh']
            else:
                raise NotImplementedError

            # N2 x 1 x (2 x lstm_hidden_dim)
            c_mot_states_last_sel_pairs = tf.expand_dims(c_mot_states_old, axis=1)
            # N2 x N1 x (2 x lstm_hidden_dim)
            c_mot_states_last_sel_pairs = tf.tile(c_mot_states_last_sel_pairs, [1, self.var['detbb_num'], 1])
            # (N2 x N1) x (2 x lstm_hidden_dim)
            c_mot_states_last_sel_pairs = tf.reshape(c_mot_states_last_sel_pairs, [-1, 2 * self.cfg.MOT_HIDDEN_DIM])

            # 1 x N1 x 1 x mot_feat_dim
            bbox_embed_last_pairs = tf.expand_dims(bbox_embed_last, axis=0)
            # N2 x N1 x 1 x mot_feat_dim
            bbox_embed_last_pairs = tf.tile(bbox_embed_last_pairs, [self.var['track_num'], 1, 1, 1])
            # (N2 x N1) x 1 x mot_feat_dim
            bbox_embed_last_pairs = tf.reshape(bbox_embed_last_pairs, [-1, 1, self.cfg.MOT_LAYER_DIM])

            with tf.control_dependencies(
                [tf.assert_equal(tf.shape(bbox_embed_last_pairs)[0], self.var['track_num'] * self.var['detbb_num']),
                tf.assert_equal(tf.shape(bbox_embed_last_pairs)[2], self.cfg.MOT_LAYER_DIM)]
            ):
                one_step = tf.reshape(tf.constant(1), [1,])
            repeat = tf.reshape(self.var['track_num'] * self.var['detbb_num'], [1,])
            # (N2 x N1)
            one_step = tf.tile(one_step, repeat)

            h_mot_states_pre_last, c_mot_states_last = tf.nn.dynamic_rnn(
                                                        lstm_cell,
                                                        bbox_embed_last_pairs,
                                                        initial_state=c_mot_states_last_sel_pairs,
                                                        sequence_length=one_step
                                                    )

            # (N2 x N1) x 1 x (2 * lstm_hidden_dim), (N2 x N1) x 1 x lstm_hidden_dim
            c_mot_states_all_tmp, h_mot_states_pre_tmp = h_mot_states_pre_last

            with tf.control_dependencies(
                [tf.assert_equal(tf.shape(c_mot_states_all_tmp)[0], self.var['track_num'] * self.var['detbb_num']),
                tf.assert_equal(tf.shape(c_mot_states_all_tmp)[1], 1)]
            ):  
                # (N2 x N1) x (2 * lstm_hidden_dim),
                c_mot_states_all_last_tmp = tf.reshape(
                                                c_mot_states_all_tmp,
                                                [self.var['track_num'] * self.var['detbb_num'], 2 * self.cfg.MOT_HIDDEN_DIM])

            with tf.control_dependencies(
                [tf.assert_equal(c_mot_states_all_last_tmp, c_mot_states_last)]
            ):            
                # N2 x N1 x 1 x lstm_hidden_dim
                h_mot_states_last = tf.reshape(h_mot_states_pre_tmp,
                                               [self.var['track_num'], self.var['detbb_num'], 1, self.cfg.MOT_HIDDEN_DIM])
            
            if self.mode == 'test':
                # N2 x (2 x lstm_hidden_dim)
                c_mot_states_test = tf.reshape(c_mot_states_old, [-1, 2 * self.cfg.MOT_HIDDEN_DIM])
                # N1 x 1 x mot_feat_dim
                bbox_embed_last_test = tf.reshape(bbox_embed_last, [-1, 1, self.cfg.MOT_LAYER_DIM])
                with tf.control_dependencies(
                    [tf.assert_equal(tf.shape(bbox_embed_last_test)[0], self.var['detbb_num']),
                     tf.assert_equal(tf.shape(c_mot_states_test)[0], self.var['track_num']),
                    # I only use this part of the code in testing where detbb_num == track_num 
                     tf.assert_equal(self.var['detbb_num'], self.var['track_num'])]
                ):
                    one_step_test = tf.reshape(tf.constant(1), [1,])
                    repeat_test = tf.reshape(self.var['track_num'], [1,])
                # N2
                one_step_test = tf.tile(one_step_test, repeat_test)

                h_mot_states_pre_last_test, c_mot_states_last_test = tf.nn.dynamic_rnn(
                                                                         lstm_cell,
                                                                         bbox_embed_last_test,
                                                                         initial_state=c_mot_states_test,
                                                                         sequence_length=one_step_test
                                                                     )
                # N2 x 1 x (2 * lstm_hidden_dim), N2 x 1 x lstm_hidden_dim
                c_mot_states_all_test_tmp, h_mot_states_pre_test_tmp = h_mot_states_pre_last_test
                # N2 x (2 * lstm_hidden_dim),
                c_mot_states_all_last_test_tmp = tf.reshape(c_mot_states_all_test_tmp, [self.var['track_num'], 2 * self.cfg.MOT_HIDDEN_DIM])
                with tf.control_dependencies([tf.assert_equal(c_mot_states_all_last_test_tmp, c_mot_states_last_test)]):
                    # N2 x 1 x lstm_hidden_dim
                    h_mot_states_last_test = tf.reshape(h_mot_states_pre_test_tmp, [self.var['track_num'], self.cfg.MOT_HIDDEN_DIM])
                self.bookkeeping['h_mot_states_test'] = h_mot_states_last_test
                self.bookkeeping['c_mot_states_last_test'] = c_mot_states_last_test

        if self.mode == 'train':
            self.bookkeeping['istate_mot_check_old'] = self.plh['istate_mot']
            self.bookkeeping['seq_len_check_old'] = seq_len
            self.bookkeeping['c_mot_states_first_sel'] = c_mot_states_first_sel
            self.bookkeeping['c_mot_states_last_sel'] = c_mot_states_last_sel
            self.bookkeeping['c_mot_states'] = c_mot_states_old
            self.bookkeeping['c_mot_states_all'] = c_mot_states_all

             # sync verification
            mot_frame_num = tf.reshape(self.plh['mot_frame_num'], [self.var['track_num'], self.plh['num_step_by_user'], 1])
            mot_frame_num = tf.cast(mot_frame_num, tf.float32)
            det_bbox_frame_minus_one = self.gather_final_states(mot_frame_num, seq_len)
            det_bbox_frame_minus_one = tf.reshape(det_bbox_frame_minus_one,[self.var['track_num']])
            detection_bbox_org = tf.reshape(self.plh['orig_noisy_bboxes_synced_with_mot'], [self.var['track_num'], self.plh['num_step_by_user'], self.cfg.MOT_INPUT_DIM])
            detection_bbox_associated_with_init_state = detection_bbox_org[:, 0, :]
            lstm_state_frame_for_verification = mot_frame_num[:, 0, :]
            self.bookkeeping['mot_det_bbox_frame_minus_one'] = det_bbox_frame_minus_one
            self.bookkeeping['mot_lstm_state_frame_for_verification'] = lstm_state_frame_for_verification
            self.bookkeeping['mot_det_bbox_associated_with_init_state'] = detection_bbox_associated_with_init_state

        self.bookkeeping['track_len'] = self.plh['track_len']
        self.bookkeeping['mid_missdet_num'] = self.plh['mid_missdet_num']
        self.bookkeeping['e_offset'] = self.plh['end_offset']
        self.bookkeeping['detbb_check1'] = bbox_embed_wo_last_aligned
        self.bookkeeping['detbb_check2'] = bbox_embed_last
        self.bookkeeping['h_mot_states'] = h_mot_states_last
        # tracker keeps this as a track memory for each track
        self.bookkeeping['c_mot_states_last'] = c_mot_states_last

        return h_mot_states_last

    def build_mot_lstm_predictor(self, hidden_states_mot):

        with tf.variable_scope("mot_lstm/output"):
            assert(int(self.cfg.APP_HIDDEN_DIM / self.cfg.APP_LAYER_DIM) == 8)
            weights_cls1 = self.init_weights('weights1', self.FLAGS.weight_decay, [self.cfg.MOT_HIDDEN_DIM, 8])
            biases_cls1 = self.init_biases('biases1', [8])        
            weights_cls2 = self.init_weights('weights2', self.FLAGS.weight_decay, [8, self.cfg.OUT_NUM])
            biases_cls2 = self.init_biases('biases2', [self.cfg.OUT_NUM])

            # (N2 x N2 x T) x lstm_hidden_dim
            assert(self.cfg.SAMPLE_NUM_FOR_LOSS == 1)

            hidden_states_mot = tf.reshape(
                                    hidden_states_mot, 
                                    [self.var['track_num'] * self.var['detbb_num'], self.cfg.MOT_HIDDEN_DIM])

            with tf.control_dependencies(
                [tf.assert_equal(tf.shape(hidden_states_mot)[1], self.cfg.MOT_HIDDEN_DIM),
                 tf.assert_equal(tf.shape(hidden_states_mot)[0], self.var['track_num'] * self.var['detbb_num'] * self.cfg.SAMPLE_NUM_FOR_LOSS)]
            ):
                # (N2 x N2 x T) x 8
                cls_prediction_tmp = tf.nn.relu(tf.matmul(hidden_states_mot, weights_cls1) + biases_cls1)
            # (N2 x N2 x T) x output_dim
            cls_prediction_vec = tf.matmul(cls_prediction_tmp, weights_cls2) + biases_cls2

            assert(self.cfg.SAMPLE_NUM_FOR_LOSS == 1)
            cls_prediction = cls_prediction_vec
            # N2 x N2 x T x output_dim
            with tf.control_dependencies(
                [tf.assert_equal(
                    tf.shape(cls_prediction_vec)[0],
                    self.var['track_num'] * self.var['detbb_num'] * self.cfg.SAMPLE_NUM_FOR_LOSS)]
            ):
                cls_prediction = tf.reshape(cls_prediction, [self.var['track_num'], self.var['detbb_num'], -1, self.cfg.OUT_NUM])

        predictions = cls_prediction, cls_prediction_vec

        return (predictions, cls_prediction_tmp)


class app_mot_gating_net(app_gating_net_BLSTM_MTP, mot_gating_net_LSTM):
    
    def __init__(self, FLAGS, cfg, mode):

        super(app_mot_gating_net, self).__init__(FLAGS, cfg, mode)
        self.plh = {}
        self.var = {}
        self.setup_plh()
        self.setup_var()

        # self.boundaries = [4446*2, 7436*2, 9024*2]
        self.boundaries = [18710*2, 28114*2, 36506*2]
        self.values = [0.1, 0.4, 0.7, 1.0]

    def setup_plh(self):

        # motion input
        self.plh['detection_bboxes'] = tf.placeholder(tf.float32, shape=(None, self.cfg.MOT_INPUT_DIM)) 
        self.plh['valid_mot_data'] = tf.placeholder(tf.float32, shape=[None, None, 1])
        self.plh['istate_mot'] = tf.placeholder(tf.float32, [None, 2 * self.cfg.MOT_HIDDEN_DIM])    
        self.plh['start_offset'] = tf.placeholder(tf.int32, shape=[None])
        self.plh['end_offset'] = tf.placeholder(tf.int32, shape=[None])
        self.plh['mid_missdet_num'] = tf.placeholder(tf.int32, shape=[None])
        self.plh['first_missdet_num'] = tf.placeholder(tf.int32, shape=[None])
        self.plh['c_mot_states_plh'] = tf.placeholder(tf.float32, [None, 2 * self.cfg.MOT_HIDDEN_DIM])

        # appearance input
        images_shape = [None, self.cfg.IMAGE_HEIGHT, self.cfg.IMAGE_WIDTH, 3]
        self.plh['images'] = tf.placeholder(tf.float32, shape=images_shape)    
        # indices for building memory that contains negative target information
        self.plh['indices_by_user'] = tf.placeholder(tf.int32, [None, None, None, 2], name='frame_sel')
        self.plh['valid_app_data'] = tf.placeholder(tf.float32, [None, None, 1], name='valid_app_data')
        self.plh['indices_for_mapping'] = tf.placeholder(tf.int32, [None, 1], name='mapping_ind')
        self.plh['image_batch_shape'] = tf.placeholder(tf.int32, [None], name='image_batch_shape')
        self.plh['sel_pool_indices'] = tf.placeholder(tf.float32, [None, None, 1])
        # memories for tracks that started earlier than the current frame
        self.plh['prev_track_memory'] = tf.placeholder(tf.float32, [None, self.cfg.APP_HIDDEN_DIM], 'prev_mem')
        self.plh['prev_track_ind'] = tf.placeholder(tf.int32, [None], name='prev_mem_ind')
        self.plh['nonprev_track_ind'] = tf.placeholder(tf.int32, [None], name='nonprev_mem_ind')

        # other input
        self.plh['is_training'] = tf.placeholder(tf.bool, [], name='is_training')
        self.plh['num_step_by_user'] = tf.placeholder(tf.int32, [], name='num_step')
        self.plh['frames_by_user'] = tf.placeholder(tf.int32, [1], name='frame_sel')
        self.plh['sel_indices']= tf.placeholder(tf.float32, [None, None, None], name='select_data')
        self.plh['track_len'] = tf.placeholder(tf.int32, [None], name='track_len')
        self.plh['track_len_offset'] = tf.placeholder(tf.int32, [None], name='track_len_offset')  

        # concated lstm states container
        self.plh['istate_app'] = tf.placeholder(tf.float32, [None, 2 * self.cfg.APP_HIDDEN_DIM])

        # needed in test mode
        self.plh['app_embed_plh'] = tf.placeholder(tf.float32, shape=[None, 1, self.cfg.APP_LAYER_DIM])
        self.plh['h_app_track_memory_plh'] = tf.placeholder(tf.float32, [None, 1, self.cfg.APP_HIDDEN_DIM])

        # new
        self.plh['orig_noisy_bboxes_synced_with_app'] = tf.placeholder(tf.float32, shape=[None, self.cfg.MOT_INPUT_DIM])
        self.plh['orig_noisy_bboxes_synced_with_mot'] = tf.placeholder(tf.float32, shape=[None, self.cfg.MOT_INPUT_DIM])
        self.plh['app_frame_num'] = tf.placeholder(tf.int32, shape=[None, None])
        self.plh['mot_frame_num'] = tf.placeholder(tf.int32, shape=[None, None])
        self.plh['trk_bbox_org_prev'] = tf.placeholder(tf.float32, shape=[None, self.cfg.MOT_INPUT_DIM])

        if self.mode == 'test':
            self.plh['detbb_num_test'] = tf.placeholder(tf.int32, [])
            self.plh['track_num_test'] = tf.placeholder(tf.int32, [])
            self.plh['trk_bbox_org'] = tf.placeholder(tf.float32, shape=[None, self.cfg.MOT_INPUT_DIM])
            self.plh['det_bbox_org'] = tf.placeholder(tf.float32, shape=[None, self.cfg.MOT_INPUT_DIM])

    def setup_var(self):

        # motion net variables    
        if self.mode == 'train':
            # other variables
            self.var['detbb_num'] = tf.shape(self.plh['images'])[0] / self.plh['num_step_by_user']
            detbb_num_tmp = tf.shape(self.plh['detection_bboxes'])[0] / self.plh['num_step_by_user']
            with tf.control_dependencies([tf.assert_equal(self.var['detbb_num'], detbb_num_tmp)]):
                self.var['track_num'] = tf.shape(self.plh['istate_app'])[0]
            self.var['num_step_check'] = tf.shape(self.plh['images'])[0] / self.var['detbb_num']
            track_num_tmp = tf.shape(self.plh['istate_mot'])[0]        
            # currently, I assume that detbb_num == track_num
            with tf.control_dependencies([tf.assert_equal(self.var['detbb_num'], track_num_tmp),
                                          tf.assert_equal(self.var['track_num'], track_num_tmp)]):
                track_num_tmp = tf.shape(self.plh['istate_mot'])[0]
        elif self.mode == 'test':
            self.var['detbb_num'] = self.plh['detbb_num_test']
            self.var['track_num'] = self.plh['track_num_test']
            track_num_tmp = self.var['track_num']
        else:
            raise NotImplementedError

        self.var['num_step'] = tf.shape(tf.reshape(
                                            self.plh['detection_bboxes'], 
                                            [self.var['detbb_num'], -1, self.cfg.MOT_INPUT_DIM]))[1]
        self.var['num_step_scalar'] = tf.reshape(self.var['num_step'], [1,])

    def predict(self):
        
        # Appearance Branch 
        self.var['app_embed'] = self.build_cnn()
        if self.mode == 'train':
            app_embed = self.var['app_embed']
            self.var['h_app_track_memory'] = self.build_lstm_track_memory(app_embed)
            self.var['h_app_track_memory_plh'] = self.var['h_app_track_memory']
        else:
            app_embed = self.plh['app_embed_plh']
            self.var['h_app_track_memory'] = self.build_lstm_track_memory(app_embed)
            self.var['h_app_track_memory_plh'] = self.plh['h_app_track_memory_plh']
        _, app_mem_final = self.build_blstm_mtp_predictor(app_embed)

        # Motion Branch 
        bbox_embed = self.compute_mot_features()
        bbox_embed_wo_last, bbox_embed_last = self.generate_matching_pairs(bbox_embed)
        hidden_states_mot = self.build_mot_lstm_memory(bbox_embed_wo_last, bbox_embed_last)
        _, mot_mem_final = self.build_mot_lstm_predictor(hidden_states_mot)

        # Appearance + Motion 
        predictions = self.build_app_mot_predictor(app_mem_final, mot_mem_final)

        return predictions
    
    def build_app_mot_predictor(self, appearance_memory, motion_memory):

        with tf.variable_scope('app_mot_predictor/merge'):

            blstm_dim = int(self.cfg.APP_HIDDEN_DIM / self.cfg.APP_LAYER_DIM)

            with tf.control_dependencies(
                [tf.assert_equal(tf.shape(appearance_memory)[0], tf.shape(motion_memory)[0])]
            ):
                cur_batch_num = tf.shape(appearance_memory)[0]
                app_mem_dim = tf.shape(appearance_memory)[1]
                mot_mem_dim = tf.shape(motion_memory)[1]
            with tf.control_dependencies(
                [tf.assert_equal(app_mem_dim, 2 * blstm_dim),
                tf.assert_equal(mot_mem_dim, tf.cast(app_mem_dim / 2, tf.int32))]
            ):             

                # 1x1 conv
                # (N2 x N1 x T) x 1 x 1 x app_mem_dim
                appearance_memory = tf.reshape(
                        appearance_memory, 
                        [cur_batch_num, 1, -1, 2 * blstm_dim])
            # (N2 x N1 x T) x 1 x 1 x mot_mem_dim                                           
            motion_memory = tf.reshape(
                    motion_memory,
                    [cur_batch_num, 1, -1, blstm_dim])

            # 1x1 conv  
            with slim.arg_scope([slim.conv2d], stride=1,
                    padding='SAME', activation_fn=tf.nn.relu):
                with tf.control_dependencies(
                    [tf.assert_equal(tf.shape(appearance_memory)[2], 1),
                     tf.assert_equal(tf.shape(motion_memory)[2], 1)]
                ):
                    appearance_memory = slim.conv2d(appearance_memory,
                                                    2 * blstm_dim,
                                                    [1, 1],
                                                    scope='app_1x1conv_1')
                appearance_memory = slim.conv2d(appearance_memory,
                                                2 * blstm_dim,
                                                [1, 1],
                                                scope='app_1x1conv_2')
                motion_memory = slim.conv2d(motion_memory,
                                            blstm_dim,
                                            [1, 1],
                                            scope='mot_1x1conv_1')
                motion_memory = slim.conv2d(motion_memory,
                                            blstm_dim,
                                            [1, 1],
                                            scope='mot_1x1conv_2')

            # (N2 x N1 x T) x app_mem_dim
            appearance_memory = tf.reshape(appearance_memory, [cur_batch_num, app_mem_dim])
            # (N2 x N1 x T) x mot_mem_dim
            motion_memory = tf.reshape(motion_memory, [cur_batch_num, mot_mem_dim])

            keep_prob = tf.compat.v1.train.piecewise_constant(self.global_step, self.boundaries, self.values)

            if self.mode == 'train':
                appearance_memory = tf.nn.dropout(appearance_memory, keep_prob=keep_prob)

            # (N2 x N1 x T) x (app_mem_dim + mot_mem_dim)
            memory = tf.concat([appearance_memory, motion_memory], axis=1)
            
        app_mot_mem_dim = 3 * blstm_dim
        with tf.variable_scope('app_mot_predictor/output'):
            with tf.control_dependencies(
                [tf.assert_equal(tf.shape(memory)[1], app_mem_dim + mot_mem_dim),
                 tf.assert_equal(tf.shape(memory)[1], app_mot_mem_dim)]
            ):            
                weights_cls1 = self.init_weights('weights1', self.FLAGS.weight_decay, [app_mot_mem_dim, app_mot_mem_dim])
                biases_cls1 = self.init_biases('biases1', [app_mot_mem_dim])  
                weights_cls2 = self.init_weights('weights2', self.FLAGS.weight_decay, [app_mot_mem_dim, self.cfg.OUT_NUM])
                biases_cls2 = self.init_biases('biases2', [self.cfg.OUT_NUM])
                # (N2 x N1 x T) x (app_mem_dim + mot_mem_dim)            
                cls_prediction_tmp = tf.nn.relu(tf.matmul(memory, weights_cls1) + biases_cls1)
                # (N2 x N1 x T) x output_dim
                cls_prediction_vec = tf.matmul(cls_prediction_tmp, weights_cls2) + biases_cls2
                # N2 x N1 x T x output_dim
                cls_prediction = tf.reshape(
                                    cls_prediction_vec,
                                    [self.var['track_num'], self.var['detbb_num'], -1, self.cfg.OUT_NUM]
                                 )

                predictions = cls_prediction, cls_prediction_vec

        return predictions

    def collect_trainer_ops(self, losses, train_op, logging_ops):
        
        bookkeeping = self.bookkeeping

        total_loss, reg_loss, cls_loss = losses
        savers = logging_ops['savers']
        train_log_ops = logging_ops['train_log_ops']
        val_log_ops = logging_ops['val_log_ops']

        tf_ops = {}
        tf_ops['train_op'] = train_op
        tf_ops['cls_loss'] = cls_loss
        tf_ops['reg_loss'] = reg_loss
        tf_ops['total_loss'] = total_loss
        tf_ops['accuracy_total'] = bookkeeping['accuracy_total']
        tf_ops['labels_vec'] = bookkeeping['labels_vec']
        tf_ops['predicted_classes'] = bookkeeping['predicted_classes']
        tf_ops['detbb_num'] = self.var['detbb_num']
        tf_ops['num_step_check'] = self.var['num_step_check']
        tf_ops['frames_by_user'] = self.plh['frames_by_user']
        tf_ops['h_app_states_first_sel'] = bookkeeping['h_app_states_first_sel']
        tf_ops['c_app_states_first_sel'] = bookkeeping['c_app_states_first_sel']
        tf_ops['c_app_states_last_sel'] = bookkeeping['c_app_states_last_sel']
        tf_ops['c_app_states'] = bookkeeping['c_app_states']
        tf_ops['c_mot_states_first_sel'] = bookkeeping['c_mot_states_first_sel']
        tf_ops['c_mot_states_last_sel'] = bookkeeping['c_mot_states_last_sel']
        tf_ops['c_mot_states'] = bookkeeping['c_mot_states']
        tf_ops['app_trk_bbox_frame'] = bookkeeping['app_trk_bbox_frame_for_verification']
        tf_ops['app_det_bbox_frame'] = bookkeeping['app_det_bbox_frame_for_verification']
        tf_ops['app_lstm_state_frame'] = bookkeeping['app_lstm_state_frame_for_verification']
        tf_ops['app_det_bbox_associated_with_init_state'] = bookkeeping['app_det_bbox_associated_with_init_state']
        tf_ops['mot_trk_bbox_frame'] = bookkeeping['mot_trk_bbox_frame_for_verification']
        tf_ops['mot_det_bbox_frame'] = bookkeeping['mot_det_bbox_frame_for_verification']
        tf_ops['mot_det_bbox_frame_minus_one'] = bookkeeping['mot_det_bbox_frame_minus_one']
        tf_ops['mot_lstm_state_frame'] = bookkeeping['mot_lstm_state_frame_for_verification']
        tf_ops['mot_det_bbox_associated_with_init_state'] = bookkeeping['mot_det_bbox_associated_with_init_state']

        train_log_ops['h_app_states_first_sel'] = bookkeeping['h_app_states_first_sel']
        train_log_ops['c_app_states_first_sel'] = bookkeeping['c_app_states_first_sel']
        train_log_ops['c_mot_states_first_sel'] = bookkeeping['c_mot_states_first_sel']
        train_log_ops['app_lstm_state_frame'] = bookkeeping['app_lstm_state_frame_for_verification']
        train_log_ops['app_det_bbox_associated_with_init_state'] = bookkeeping['app_det_bbox_associated_with_init_state']
        train_log_ops['mot_lstm_state_frame'] = bookkeeping['mot_lstm_state_frame_for_verification']
        train_log_ops['mot_det_bbox_associated_with_init_state'] = bookkeeping['mot_det_bbox_associated_with_init_state']
        val_log_ops['h_app_states_first_sel'] = bookkeeping['h_app_states_first_sel']
        val_log_ops['c_app_states_first_sel'] = bookkeeping['c_app_states_first_sel']
        val_log_ops['c_mot_states_first_sel'] = bookkeeping['c_mot_states_first_sel']
        val_log_ops['app_lstm_state_frame'] = bookkeeping['app_lstm_state_frame_for_verification']
        val_log_ops['app_det_bbox_associated_with_init_state'] = bookkeeping['app_det_bbox_associated_with_init_state']
        val_log_ops['mot_lstm_state_frame'] = bookkeeping['mot_lstm_state_frame_for_verification']
        val_log_ops['mot_det_bbox_associated_with_init_state'] = bookkeeping['mot_det_bbox_associated_with_init_state']
       
        tf_placeholders = {}
        tf_placeholders['images'] = self.plh['images']
        tf_placeholders['istate_app'] = self.plh['istate_app']
        tf_placeholders['is_training'] = self.plh['is_training']
        tf_placeholders['num_step_by_user'] = self.plh['num_step_by_user']
        tf_placeholders['frames_by_user'] = self.plh['frames_by_user']
        tf_placeholders['indices_by_user'] = self.plh['indices_by_user']
        tf_placeholders['track_len'] = self.plh['track_len']
        tf_placeholders['sel_indices'] = self.plh['sel_indices']
        tf_placeholders['valid_app_data'] = self.plh['valid_app_data']
        tf_placeholders['indices_for_mapping'] = self.plh['indices_for_mapping']
        tf_placeholders['image_batch_shape'] = self.plh['image_batch_shape']
        tf_placeholders['track_len_offset'] = self.plh['track_len_offset']
        tf_placeholders['sel_pool_indices'] = self.plh['sel_pool_indices']
        tf_placeholders['prev_track_memory'] = self.plh['prev_track_memory']
        tf_placeholders['prev_track_ind'] = self.plh['prev_track_ind']
        tf_placeholders['nonprev_track_ind'] = self.plh['nonprev_track_ind']
        tf_placeholders['detection_bboxes'] = self.plh['detection_bboxes']
        tf_placeholders['valid_mot_data'] = self.plh['valid_mot_data']
        tf_placeholders['start_offset'] = self.plh['start_offset']
        tf_placeholders['end_offset'] = self.plh['end_offset']
        tf_placeholders['istate_mot'] = self.plh['istate_mot']
        tf_placeholders['mid_missdet_num'] = self.plh['mid_missdet_num']
        tf_placeholders['first_missdet_num'] = self.plh['first_missdet_num']    
        tf_placeholders['orig_noisy_bboxes_synced_with_app'] = self.plh['orig_noisy_bboxes_synced_with_app']
        tf_placeholders['orig_noisy_bboxes_synced_with_mot'] = self.plh['orig_noisy_bboxes_synced_with_mot']
        tf_placeholders['app_frame_num'] = self.plh['app_frame_num']
        tf_placeholders['mot_frame_num'] = self.plh['mot_frame_num']
        tf_placeholders['trk_bbox_org_prev']  = self.plh['trk_bbox_org_prev']

        tf_vars = [tf_ops, train_log_ops, val_log_ops, tf_placeholders]

        return (tf_vars, savers)

    def collect_tracker_ops(self, cls_prediction_vec):
        
        # (N2 * N1 * T) x output_dim
        softmax_out = tf.nn.softmax(cls_prediction_vec)

        tf_logging_ops = self.get_logging_op()
        savers = tf_logging_ops['savers']

        tf_ops = {}
        tf_ops['softmax_out'] = softmax_out
        tf_ops['h_app_states'] = self.bookkeeping['h_app_states']
        tf_ops['c_app_states'] = self.bookkeeping['c_app_states']
        tf_ops['h_app_track_memory'] = self.var['h_app_track_memory']
        tf_ops['m_states'] = self.bookkeeping['m_states']
        tf_ops['app_embed'] = self.var['app_embed']
        tf_ops['c_mot_states_last'] = self.bookkeeping['c_mot_states_last']
        tf_ops['h_mot_states'] = self.bookkeeping['h_mot_states']
        tf_ops['c_mot_states_last_test'] = self.bookkeeping['c_mot_states_last_test']
        tf_ops['h_mot_states_test'] = self.bookkeeping['h_mot_states_test']

        tf_placeholders = {}
        tf_placeholders['images'] = self.plh['images']
        tf_placeholders['istate_app'] = self.plh['istate_app']
        tf_placeholders['is_training'] = self.plh['is_training']
        tf_placeholders['num_step_by_user'] = self.plh['num_step_by_user']
        tf_placeholders['frames_by_user'] = self.plh['frames_by_user']
        tf_placeholders['indices_by_user'] = self.plh['indices_by_user']
        tf_placeholders['sel_pool_indices'] = self.plh['sel_pool_indices']
        tf_placeholders['h_app_track_memory_plh'] = self.plh['h_app_track_memory_plh']
        tf_placeholders['track_len'] = self.plh['track_len']
        tf_placeholders['app_embed_plh'] = self.plh['app_embed_plh']
        tf_placeholders['valid_app_data'] = self.plh['valid_app_data']
        tf_placeholders['indices_for_mapping'] = self.plh['indices_for_mapping']
        tf_placeholders['image_batch_shape'] = self.plh['image_batch_shape']
        tf_placeholders['track_len_offset'] = self.plh['track_len_offset']
        tf_placeholders['detection_bboxes'] = self.plh['detection_bboxes']
        tf_placeholders['valid_mot_data'] = self.plh['valid_mot_data']
        tf_placeholders['start_offset'] = self.plh['start_offset']
        tf_placeholders['end_offset'] = self.plh['end_offset']
        tf_placeholders['istate_mot'] = self.plh['istate_mot']
        tf_placeholders['c_mot_states_plh'] = self.plh['c_mot_states_plh']
        tf_placeholders['mid_missdet_num'] = self.plh['mid_missdet_num']
        tf_placeholders['first_missdet_num'] = self.plh['first_missdet_num']
        tf_placeholders['det_bbox_org'] = self.plh['det_bbox_org']
        tf_placeholders['trk_bbox_org'] = self.plh['trk_bbox_org']
        tf_placeholders['app_frame_num'] = self.plh['app_frame_num']
        tf_placeholders['sel_indices'] = self.plh['sel_indices'] 
        tf_placeholders['detbb_num'] = self.plh['detbb_num_test']
        tf_placeholders['track_num'] = self.plh['track_num_test']

        tf_vars = [tf_ops, tf_placeholders]

        return (tf_vars, savers)

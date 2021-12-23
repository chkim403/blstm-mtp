import numpy as np
import tensorflow as tf
from collections import defaultdict

class Greedy_Tracker(object):

    def __init__(self, cfg_tracker, cfg_train, tf_ops, tf_placeholders, session):

        self.network_type = cfg_tracker.network_type
        self.cls_thr = cfg_tracker.nn_gating_thr
        self.det_ratio_thr = cfg_tracker.det_ratio
        self.N_miss_max = cfg_tracker.N_miss_max
        self.img_height = cfg_tracker.IMAGE_HEIGHT
        self.img_width = cfg_tracker.IMAGE_WIDTH
        self.all_tracks = defaultdict(lambda: defaultdict(defaultdict))
        self.track_num = 0
        self.model_info = {}
        self.model_info['app_hidden_dim'] = cfg_train.APP_HIDDEN_DIM
        self.model_info['mot_hidden_dim'] = cfg_train.MOT_HIDDEN_DIM
        self.model_info['mot_input_dim'] = cfg_train.MOT_INPUT_DIM
        self.result = []
        
        self.cfg_train = cfg_train
        self.cfg_tracker = cfg_tracker

        self.sess = session
        self.tf_ops = tf_ops
        self.tf_plh = tf_placeholders
       
        self.neg_mem_indices = self.precompute_neg_mem_indices()        

    def precompute_neg_mem_indices(self):
        # get indices for online negative examples (i.e. other tracks in the scene) for each track

        # NOTE: need to be set again when the code is used for tracking more objects
        max_track_num = 200
        max_det_num = 200

        neg_mem_ind = np.zeros((max_track_num, max_det_num, max_track_num-1, 2))
        for i in range(100):
            for j in range(100):
                xy_ind_tmp = np.zeros((max_track_num - 1, 2))
                x_ind_tmp = np.arange(max_track_num, dtype=np.int32)
                xy_ind_tmp[:, 0] = x_ind_tmp[x_ind_tmp != i]
                xy_ind_tmp[:, 1] = j
                neg_mem_ind[i, j, :, :] = xy_ind_tmp

        return neg_mem_ind

    def build_neg_mem_indices(self, track_num, det_num):

        if track_num > 1:
            neg_mem_inds = self.neg_mem_indices[:track_num, :det_num, :(track_num-1), :]
        elif track_num == 1:
            neg_mem_inds = None
        else:
            raise NotImplementedError
        
        return neg_mem_inds

    def get_lstm_states(self, h_np, c_np, cur_detbb_num, is_track_state):

        h_np = np.reshape(h_np, (cur_detbb_num, cur_detbb_num, -1))
        c_np = np.reshape(c_np, (cur_detbb_num, cur_detbb_num, -1))
        if is_track_state == True:
            h_np = np.transpose(h_np, (1, 0, 2))
            c_np = np.transpose(c_np, (1, 0, 2))

        # loop can be commented out later to improve processing time
        # check lstm states
        h_np = np.reshape(h_np , (cur_detbb_num * cur_detbb_num, -1))
        for kkk in range(1, cur_detbb_num):
            assert(np.array_equal(h_np[kkk*cur_detbb_num:(kkk+1)*cur_detbb_num, :], \
                            h_np[:cur_detbb_num, :]))
        h_np = h_np[:cur_detbb_num, :]

        # check lstm states
        c_np = np.reshape(c_np , (cur_detbb_num * cur_detbb_num, -1))
        for kkk in range(1, cur_detbb_num):
            assert(np.array_equal(c_np[kkk*cur_detbb_num:(kkk+1)*cur_detbb_num, :], \
                            c_np[:cur_detbb_num, :]))
        c_np = c_np[:cur_detbb_num, :]

        return (h_np, c_np)

    def get_lstm_states_new(self, h_np, c_np, cur_detbb_num):

        h_np = np.reshape(h_np, (cur_detbb_num, -1))
        c_np = np.reshape(c_np, (cur_detbb_num, -1))

        h_np = h_np[:cur_detbb_num, :]
        c_np = c_np[:cur_detbb_num, :]

        return (h_np, c_np)

    def get_lstm_states_for_matched_tracks(self, matching, model_dim, h_np, c_np, trk_num, det_num):

        inds_sel1 = []
        track_i_sel = []
        # select lstm states for matched tracks
        if len(matching) > 0:
            h_np_tmp = np.zeros((len(matching), model_dim))
            c_np_tmp = np.zeros((len(matching), 2 * model_dim))
            h_np = np.reshape(h_np, (trk_num, det_num, -1))     
            c_np = np.reshape(c_np, (trk_num, det_num, -1))                             
            for kkk in range(0, len(matching)):
                track_i = int(matching[kkk][0, 0])
                detbb_i = int(matching[kkk][0, 1])
                h_np_tmp[kkk, :] = h_np[track_i, detbb_i, :]
                c_np_tmp[kkk, :] = c_np[track_i, detbb_i, :]
                inds_sel1.append(detbb_i)
                track_i_sel.append(track_i)
            h_np = h_np_tmp
            c_np = c_np_tmp
        else:
            h_np = []
            c_np = []
        
        return (h_np, c_np, inds_sel1, track_i_sel)

    def precompute_app_features(self, imgs, bbs):

        cur_detbb_num = np.shape(imgs)[0]
        assert(cur_detbb_num == np.shape(bbs)[0])

        feed_dict = {
            self.tf_plh['detbb_num']: cur_detbb_num,
            self.tf_plh['images']:imgs, 
            self.tf_plh['is_training']: False,
            self.tf_plh['num_step_by_user']: 1,
            self.tf_plh['valid_app_data']: np.ones((cur_detbb_num, 1, 1), dtype=np.int32),
            self.tf_plh['indices_for_mapping']: np.reshape(np.arange(cur_detbb_num * 1, dtype=np.int32), (-1, 1)),
            self.tf_plh['image_batch_shape']: np.array([cur_detbb_num * 1, self.cfg_train.APP_LAYER_DIM])
        }

        app_embed_np = self.sess.run(self.tf_ops['app_embed'], feed_dict=feed_dict)

        return app_embed_np

    def initialize_tracks(
            self,
            h,
            c,
            memory,
            bbs,
            bbs_norm,
            det_ids,
            frame,
            hidden_dim,
            is_dummy,
            network
        ):

        h = np.reshape(h, (-1, hidden_dim))
        if network == 'app_blstm':
            assert(np.shape(memory)[0] == np.shape(h)[0])
            assert(np.shape(memory)[0] == np.shape(c)[0])
        assert(np.array_equal(h, c[:, hidden_dim:]))
        assert(np.shape(h)[0] == np.shape(c)[0])

        if is_dummy == False:
            for i in range(0, np.shape(h)[0]):
                self.track_num += 1
                # 1 x d
                self.all_tracks[self.track_num]['h_states'] = h[i, :]
                # 1 x d
                self.all_tracks[self.track_num]['c_states'] = c[i, :]
                self.all_tracks[self.track_num]['real_det_num'] = 1
                self.all_tracks[self.track_num]['miss_det_num'] = 0
                self.all_tracks[self.track_num]['last_miss_det_num'] = 0
                self.all_tracks[self.track_num]['bb'] = bbs[det_ids[i], :]
                self.all_tracks[self.track_num]['bb_norm'] = bbs_norm[det_ids[i], :]
                self.all_tracks[self.track_num]['frame'] = frame
                self.all_tracks[self.track_num]['th'] = [self.cls_thr]

                if network == 'app_blstm':
                    # 1 x 1 x d
                    self.all_tracks[self.track_num]['mem'] = memory[i, :, :]
                
                self.result.append((frame, det_ids[i], 1.0, self.track_num))
        elif is_dummy == True:
            ct = -1
            for i in range(0, np.shape(memory)[0]):
                ct -= 1
                # 1 x d
                self.all_tracks[ct]['h_states'] = h[i, :]
                # 1 x d
                self.all_tracks[ct]['c_states'] = c[i, :]
                self.all_tracks[ct]['real_det_num'] = 1
                self.all_tracks[ct]['miss_det_num'] = 0
                self.all_tracks[ct]['last_miss_det_num'] = 0
                self.all_tracks[ct]['bb'] = bbs[det_ids[i], :]
                self.all_tracks[ct]['bb_norm'] = bbs_norm[det_ids[i], :]
                self.all_tracks[ct]['frame'] = frame
                self.all_tracks[ct]['th'] = [self.cls_thr]
                
                if network == 'app_blstm':
                    # 1 x 1 x d
                    self.all_tracks[ct]['mem'] = memory[i, :, :]
        else:
            raise NotImplementedError

    def delete_dummy_tracks(self, frame):

        for i in self.all_tracks.keys():
            if i < 0:
                del self.all_tracks[i]

        for i in self.all_tracks.keys():
            assert(i > 0)

    def update_tracks(
            self,
            h,
            c,
            memory,
            bbs,
            bbs_norm, 
            track_ids,
            matching,
            matching_score,
            frame,
            hidden_dim,
            network,
            missdet_tracks
        ):

        h = np.reshape(h, (-1, hidden_dim))
        if np.shape(c)[0] != 0:
            if network == 'app_blstm':
                assert((np.shape(memory)[0] == np.shape(h)[0]))
                assert((np.shape(memory)[0] == np.shape(c)[0]))
            assert(np.array_equal(h, c[:, hidden_dim:]))
        assert(len(matching) == len(matching_score))

        track_ids_sel1 = []
        for i in range(0, len(matching)):   
            track_i = int(matching[i][0, 0])
            detbb_i = int(matching[i][0, 1])
            if network == 'app_blstm':
                self.all_tracks[track_ids[track_i]]['mem'] = memory[i, :, :]
            self.all_tracks[track_ids[track_i]]['h_states'] = h[i, :]
            self.all_tracks[track_ids[track_i]]['c_states'] = c[i, :]            
            self.all_tracks[track_ids[track_i]]['real_det_num'] += 1
            self.all_tracks[track_ids[track_i]]['last_miss_det_num'] = 0
            self.all_tracks[track_ids[track_i]]['bb'] = bbs[detbb_i, :]
            self.all_tracks[track_ids[track_i]]['bb_norm'] = bbs_norm[detbb_i, :]
            self.all_tracks[track_ids[track_i]]['frame'] = frame
            self.all_tracks[track_ids[track_i]]['th'] = self.all_tracks[track_ids[track_i]]['th'] \
                                                        + [matching_score[i]]
            self.result.append((frame, detbb_i, 1.0, track_ids[track_i]))
            track_ids_sel1.append(track_ids[track_i])
                     
        # update non matched tracks with dummy detections
        track_ids_sel2 = np.setdiff1d(track_ids, track_ids_sel1)

        if network == 'mot_lstm' and len(track_ids_sel2) > 0:
            assert(np.array_equal(track_ids_sel2, missdet_tracks['track_ids']))

        for i in range(0, len(track_ids_sel2)):
            # skip dummy track
            if track_ids_sel2[i] < 0:
                continue
            self.all_tracks[track_ids_sel2[i]]['miss_det_num'] += 1
            self.all_tracks[track_ids_sel2[i]]['last_miss_det_num'] += 1
            self.result.append((frame, None, None, track_ids_sel2[i]))

            if network == 'mot_lstm' and len(track_ids_sel2) > 0:
                self.all_tracks[track_ids_sel2[i]]['h_states'] = missdet_tracks['h_states'][i, :]
                self.all_tracks[track_ids_sel2[i]]['c_states'] = missdet_tracks['c_states'][i, :]
                assert(track_ids_sel2[i] == missdet_tracks['track_ids'][i])

    def compute_iou(self, bb_p, bb_n):

        bb_px_min = bb_p[0]
        bb_py_min = bb_p[1]
        bb_pw = bb_p[2]
        bb_ph = bb_p[3]        
        bb_px_max = bb_px_min + bb_pw
        bb_py_max = bb_py_min + bb_ph

        bb_nx_min = bb_n[0]
        bb_ny_min = bb_n[1]
        bb_nw = bb_n[2]
        bb_nh = bb_n[3]    
        bb_nx_max = bb_nx_min + bb_nw
        bb_ny_max = bb_ny_min + bb_nh
        
        bb_p_area = (bb_px_max - bb_px_min)*(bb_py_max - bb_py_min)
        bb_n_area = (bb_nx_max - bb_nx_min)*(bb_ny_max - bb_ny_min)

        x1 = np.maximum(bb_px_min, bb_nx_min)
        y1 = np.maximum(bb_py_min, bb_ny_min)
        x2 = np.minimum(bb_px_max, bb_nx_max)
        y2 = np.minimum(bb_py_max, bb_ny_max)

        w = np.maximum(0.0, x2 - x1)
        h = np.maximum(0.0, y2 - y1)

        intersection = np.multiply(w, h)
        union = np.add(bb_p_area, bb_n_area) - intersection
        IoU = np.divide(intersection, union)

        return IoU

    def solve_greedy_matching(self, softmax, m_states, track_num, detbb_num, track_ids, bbs, frame):

        col1 = np.arange(track_num)        
        col2 = np.arange(detbb_num)
        col1 = np.expand_dims(col1, axis=1)
        col2 = np.expand_dims(col2, axis=0)
        col1 = np.reshape(np.tile(col1, (1, detbb_num)), (-1, 1))
        col2 = np.reshape(np.tile(col2, (track_num, 1)), (-1, 1))
        track_detbb_pair_ind = np.concatenate((col1, col2), axis=1)        
        assert(np.shape(track_detbb_pair_ind)[0] == track_num * detbb_num)

        motion_gating_mask = np.ones((track_num, detbb_num, 1))
        if self.cfg_tracker.IS_NAIVE_GATING_ON == True:     
            for i in range(0, track_num):            
                bb_p = self.all_tracks[track_ids[i]]['bb']
                bb_n = bbs

                if track_ids[i] < 0:
                    motion_gating_mask[i, :, 0] = 0
                else:
                    fr_diff = (frame - self.all_tracks[track_ids[i]]['frame'])
                    motion_gating_mask[i, :, 0] = self.naive_motion_gating(bb_p, bb_n, fr_diff)            
        motion_gating_mask = np.reshape(motion_gating_mask, (track_num * detbb_num, 1))      

        # (N1 * N2) x 1
        softmax_pos = softmax[:, 1] 
        softmax_pos = np.reshape(softmax_pos, (-1, 1))
        softmax_pos_org = softmax_pos
        softmax_pos = np.multiply(softmax_pos, motion_gating_mask)
        matching = []
        matching_score = []

        while True:
            max_p = np.amax(softmax_pos, axis=0)
            max_i = np.argmax(softmax_pos, axis=0)

            assert(softmax_pos[max_i] == max_p)
            assert(np.shape(softmax_pos)[0] == np.shape(track_detbb_pair_ind)[0])

            if max_p > self.cls_thr:
                matching.append(track_detbb_pair_ind[max_i, :])
                matching_score.append(softmax_pos_org[max_i])
                del_ind1 = track_detbb_pair_ind[:, 1] == track_detbb_pair_ind[max_i, 1]
                del_ind2 = track_detbb_pair_ind[:, 0] == track_detbb_pair_ind[max_i, 0]
                del_ind = np.where(np.logical_or(del_ind1, del_ind2))[0]

                track_detbb_pair_ind_tmp = np.delete(track_detbb_pair_ind, del_ind, axis=0)
                softmax_pos = np.delete(softmax_pos, del_ind, axis=0)
                softmax_pos_org = np.delete(softmax_pos_org, del_ind, axis=0)
                assert(len(np.where(track_detbb_pair_ind_tmp[:, 1] == track_detbb_pair_ind[max_i, 1])[0]) == 0)
                assert(len(np.where(track_detbb_pair_ind_tmp[:, 0] == track_detbb_pair_ind[max_i, 0])[0]) == 0)
                track_detbb_pair_ind = track_detbb_pair_ind_tmp
            # out of the loop when there is no good match left
            else:
                break

            # out of the loop when all detections are taken
            if np.shape(track_detbb_pair_ind)[0] == 0:
                break

        return (matching, matching_score)

    def pick_imgs(self, imgs, imgs_inds):

        imgs_sel = np.zeros((len(imgs_inds), self.img_height, self.img_width, 3))
        for i in range(0, len(imgs_inds)):
            imgs_sel[i, :, :, :] = imgs[imgs_inds[i], :, :, :]

        return imgs_sel

    def pick_dets(self, dets, dets_inds):

        dets_sel = np.zeros((len(dets_inds), self.model_info['mot_input_dim']))
        for i in range(0, len(dets_inds)):
            dets_sel[i, :] = dets[dets_inds[i], :]
        
        return dets_sel
    
    def get_gating_result(self, x_diff, y_diff, w_diff, h_diff, gating_factor):

        # NOTE: These parameters are tuned for the MOT Challenge datasets. 
        x_diff_th = 3.5
        y_diff_th = 2.0
        w_diff_th = 1.8
        h_diff_th = 1.8

        return np.logical_and(np.logical_and(x_diff < x_diff_th, y_diff < y_diff_th),
                              np.logical_and(w_diff < w_diff_th, h_diff < h_diff_th))

    def naive_motion_gating(self, bb_p, bb_n, gating_factor):

        bb_px = bb_p[0]
        bb_py = bb_p[1]
        bb_pw = bb_p[2]
        bb_ph = bb_p[3]        

        bb_nx = bb_n[:, 0]
        bb_ny = bb_n[:, 1]
        bb_nw = bb_n[:, 2]
        bb_nh = bb_n[:, 3]        

        x_diff = np.divide(np.abs(bb_px - bb_nx), bb_pw)
        y_diff = np.divide(np.abs(bb_py - bb_ny), bb_ph)
        w_diff = np.maximum(np.divide(bb_pw, bb_nw), np.divide(bb_nw, bb_pw))
        h_diff = np.maximum(np.divide(bb_ph, bb_nh), np.divide(bb_nh, bb_ph))

        return self.get_gating_result(x_diff, y_diff, w_diff, h_diff, gating_factor)

    def get_result(self):

        return self.result


class Greedy_Tracker_APP_BLSTM(Greedy_Tracker):

    def __init__(self, cfg_tracker, cfg_train, tf_ops, tf_placeholders, session):

        super(Greedy_Tracker_APP_BLSTM, self).__init__(cfg_tracker, cfg_train, tf_ops, tf_placeholders, session)

    def run(self, bbs, bbs_norm, imgs, frame_num):

        # first frame
        if len(self.all_tracks.keys()) == 0 and imgs is not None:
            
            mem_np = self.initialize_track_mems(imgs, bbs)
            h_np, c_np, memory_np = mem_np
            
            cur_detbb_num = np.shape(imgs)[0]
            self.initialize_tracks(
                h_np, 
                c_np,
                memory_np,
                bbs,
                bbs_norm,
                np.array(range(cur_detbb_num)),
                frame_num,
                self.model_info['app_hidden_dim'],
                is_dummy=False,
                network='app_blstm'
            )
            
        elif len(self.all_tracks.keys()) != 0:
            
            bookkeeping = {}
            self.data_association(imgs, bbs, bbs_norm, frame_num, bookkeeping)
            self.update_existing_tracks(bbs, bbs_norm, frame_num, bookkeeping)
            self.start_new_tracks(imgs, bbs, bbs_norm, frame_num, bookkeeping)

    def initialize_track_mems(self, imgs, bbs):       

        cur_detbb_num = np.shape(imgs)[0]
        assert(cur_detbb_num == np.shape(bbs)[0])

        # cnn features (input to lstm)
        app_embed_np = self.precompute_app_features(imgs, bbs)
        # current lstm states
        c_np = np.zeros((cur_detbb_num, 2 * self.model_info['app_hidden_dim']))

        track_mems = self.update_lstm_mems(app_embed_np, c_np, cur_detbb_num)
        h_np, c_np, memory_np = track_mems        

        return (h_np, c_np, memory_np)

    def update_lstm_mems(self, app_embed_np, c_np, cur_detbb_num):

        feed_dict = {
            self.tf_plh['track_num']: cur_detbb_num,
            self.tf_plh['app_embed_plh']: app_embed_np,
            self.tf_plh['istate_app']: c_np, 
            self.tf_plh['is_training']: False,
            self.tf_plh['num_step_by_user']: 1,
            self.tf_plh['frames_by_user']: np.array([1]),
            self.tf_plh['track_len']: np.ones(cur_detbb_num, dtype=np.int32),
            self.tf_plh['track_len_offset']: np.zeros(cur_detbb_num, dtype=np.int32),
        }

        tf_ops = [
            self.tf_ops['h_app_states'],
            self.tf_ops['c_app_states'],
            self.tf_ops['h_app_track_memory']
        ]

        # N2 x 1 x d
        h_np, c_np, memory_np = self.sess.run(tf_ops, feed_dict=feed_dict)

        return (h_np, c_np, memory_np)

    def data_association(self, imgs, bbs, bbs_norm, frame_num, bookkeeping):

        # 1. compute appearance features for new detections
        app_embed_np = []
        if imgs is not None:
            app_embed_np = self.precompute_app_features(imgs, bbs)

        # 2. load exisiting tracks
        track_memory_sel, track_bb_sel, track_ids_sel = self.collect_track_memory()
        assert(np.shape(track_memory_sel)[0] == len(track_ids_sel))

        # 3. solve matching between detections and tracks
        matching = []
        matching_score = []
        inds_sel1 = []
        istates_sel = []
        if imgs is not None and np.shape(track_memory_sel)[0] > 1:
            detbb_num = np.shape(imgs)[0]
            assert(detbb_num == np.shape(bbs)[0])
            track_num = np.shape(track_memory_sel)[0]
            softmax_np, m_states_np = self.compute_softmax_score(
                                          bbs,
                                          track_bb_sel,
                                          imgs,
                                          app_embed_np,
                                          track_memory_sel,
                                          track_ids_sel,
                                          track_num,
                                          detbb_num
                                      )

            matching, matching_score = self.solve_greedy_matching(
                                           softmax_np,
                                           m_states_np,
                                           track_num,
                                           detbb_num,
                                           track_ids_sel,
                                           bbs,
                                           frame_num
                                       )

            inds_sel1, istates_sel = self.pick_imgs_and_istates(
                                         track_ids_sel,
                                         matching,
                                         self.model_info['app_hidden_dim'],
                                     )

        bookkeeping['matching'] = matching
        bookkeeping['matching_score'] = matching_score
        bookkeeping['inds_sel1'] = inds_sel1
        bookkeeping['istates_sel'] = istates_sel
        bookkeeping['track_ids_sel'] = track_ids_sel
        bookkeeping['app_embed_np'] = app_embed_np

    def collect_track_memory(self):

        all_ids = sorted(self.all_tracks.keys())

        sel_ids = []
        for i in range(0, len(all_ids)):
            total_num = self.all_tracks[all_ids[i]]['real_det_num'] + \
                        self.all_tracks[all_ids[i]]['miss_det_num']

            if self.all_tracks[all_ids[i]]['real_det_num'] > 0:
                if (((float(self.all_tracks[all_ids[i]]['real_det_num']) / total_num) >= self.det_ratio_thr) and
                    (self.all_tracks[all_ids[i]]['last_miss_det_num'] <= self.N_miss_max)):
                    sel_ids.append(all_ids[i])
            else:
                assert((float(self.all_tracks[all_ids[i]]['real_det_num']) / total_num) < 1.0)

        track_mem_sel = np.zeros((len(sel_ids), 1, self.model_info['app_hidden_dim']))
        track_bb_sel = np.zeros((len(sel_ids), self.cfg_train.MOT_INPUT_DIM))
        for i in range(0, len(sel_ids)):
            track_mem_sel[i, :, :] = self.all_tracks[sel_ids[i]]['mem']
            track_bb_sel[i, :] = self.all_tracks[sel_ids[i]]['bb'][:self.cfg_train.MOT_INPUT_DIM]
            # track_bb_sel[i, :] = self.all_tracks[sel_ids[i]]['bb_norm'][:6]
            assert(np.shape(self.all_tracks[sel_ids[i]]['bb'])[0] == (self.cfg_train.MOT_INPUT_DIM + 1))

        return (track_mem_sel, track_bb_sel, sel_ids)

    def compute_softmax_score(self, det_bbs, trk_bbs, imgs, app_embed, trk_memory_sel, trk_ids_sel, trk_num, det_num):

        indices = self.build_neg_mem_indices(trk_num, det_num)

        # istate_app serves as dummy variables here. 
        # It does not effect the results here
        feed_dict = {
            self.tf_plh['detbb_num']: det_num,
            self.tf_plh['track_num']: trk_num,
            self.tf_plh['app_embed_plh']: app_embed,
            self.tf_plh['istate_app']: np.zeros((trk_num, 2 * self.model_info['app_hidden_dim'])),
            self.tf_plh['h_app_track_memory_plh']: trk_memory_sel,
            self.tf_plh['is_training']: False,
            self.tf_plh['num_step_by_user']: 1,
            self.tf_plh['frames_by_user']: np.array([0]),
            self.tf_plh['indices_by_user']: indices,
            self.tf_plh['track_len']: np.ones(np.shape(app_embed)[0], dtype=np.int32),
            self.tf_plh['track_len_offset']: np.zeros(np.shape(app_embed)[0], dtype=np.int32),
            self.tf_plh['valid_app_data']: np.ones((det_num, 1, 1), dtype=np.int32),
            self.tf_plh['indices_for_mapping']: np.reshape(np.arange(det_num * 1, dtype=np.int32), (-1, 1)),
            self.tf_plh['image_batch_shape']: np.array([det_num * 1, self.cfg_train.APP_LAYER_DIM]),
            self.tf_plh['det_bbox_org']: det_bbs[:, :self.cfg_train.MOT_INPUT_DIM],
            self.tf_plh['trk_bbox_org']: trk_bbs,
            self.tf_plh['app_frame_num']: np.zeros((trk_num, 1)),
            self.tf_plh['sel_indices']: np.ones((trk_num, det_num, 1))
        }

        tf_ops = [
            self.tf_ops['softmax_out'],
            self.tf_ops['m_states']
        ]

        # (N2 * N1) x 2, (N2 * N1) x (2 * (d / feat_dim)) matrix
        softmax_np, m_states_np = self.sess.run(tf_ops, feed_dict=feed_dict)

        return softmax_np, m_states_np        

    def pick_imgs_and_istates(self, track_ids, matching, hidden_dim):

        h_states_sel = np.zeros((len(matching), hidden_dim))
        c_states_sel = np.zeros((len(matching), 2 * hidden_dim))
        imgs_sel_inds = []
        for i in range(0, len(matching)):
            track_i = int(matching[i][0, 0])
            detbb_i = int(matching[i][0, 1])
            h_states_sel[i, :] = self.all_tracks[track_ids[track_i]]['h_states']
            c_states_sel[i, :] = self.all_tracks[track_ids[track_i]]['c_states']
            imgs_sel_inds.append(detbb_i)

        return (imgs_sel_inds, c_states_sel)

    def update_existing_tracks(self, bbs, bbs_norm, frame_num, bookkeeping):

        # update tracks with selected detections
        assert(len(bookkeeping['matching']) == len(bookkeeping['inds_sel1']))
        h_np = []
        c_np = []
        memory_np = []
        if len(bookkeeping['inds_sel1']) > 0:
            track_num = np.shape(bookkeeping['istates_sel'])[0]
            app_embed_sel_np = bookkeeping['app_embed_np'][bookkeeping['inds_sel1'], :]

            track_mems = self.update_lstm_mems(
                             app_embed_sel_np,
                             bookkeeping['istates_sel'],
                             track_num
                         )
            h_np, c_np, memory_np = track_mems                  

        self.update_tracks(
            h_np,
            c_np,
            memory_np,
            bbs,
            bbs_norm,
            bookkeeping['track_ids_sel'],
            bookkeeping['matching'],
            bookkeeping['matching_score'],
            frame_num,
            self.model_info['app_hidden_dim'],
            network='app_blstm',
            missdet_tracks=None
        )

        self.delete_dummy_tracks(frame_num)

    def start_new_tracks(self, imgs, bbs, bbs_norm, frame_num, bookkeeping):

        # start new tracks from detections which are not selected
        if imgs is not None:                                
            all_inds = range(np.shape(imgs)[0])
            inds_sel2 = np.setdiff1d(all_inds, bookkeeping['inds_sel1'])
            if len(inds_sel2) > 0:
                cur_detbb_num = len(inds_sel2)
                app_embed_sel2_np = bookkeeping['app_embed_np'][inds_sel2, :]

                assert(cur_detbb_num == len(inds_sel2))
                c_np = np.zeros((cur_detbb_num, 2 * self.model_info['app_hidden_dim']))       
                track_mems = self.update_lstm_mems(app_embed_sel2_np, c_np, cur_detbb_num)
                h_np, c_np, memory_np = track_mems                
    
                self.initialize_tracks(
                    h_np,
                    c_np,
                    memory_np,
                    bbs,
                    bbs_norm,
                    inds_sel2,
                    frame_num,
                    self.model_info['app_hidden_dim'],
                    is_dummy=False,
                    network='app_blstm'
                )


class Greedy_Tracker_MOT_LSTM(Greedy_Tracker):

    def __init__(self, cfg_tracker, cfg_train, tf_ops, tf_placeholders, session):

        super(Greedy_Tracker_MOT_LSTM, self).__init__(cfg_tracker, cfg_train, tf_ops, tf_placeholders, session)

    def run(self, bbs, bbs_norm, imgs, frame_num):
  
        # first frame
        if len(self.all_tracks.keys()) == 0 and imgs is not None:

            h_np, c_np = self.initialize_track_mems(imgs, bbs, bbs_norm)

            cur_detbb_num = np.shape(imgs)[0]
            self.initialize_tracks(
                h_np, 
                c_np,
                "",
                bbs,
                bbs_norm,
                np.array(range(cur_detbb_num)),
                frame_num,
                self.model_info['mot_hidden_dim'],
                is_dummy=False,
                network='mot_lstm'
            )

        elif len(self.all_tracks.keys()) != 0:

            bookkeeping = {}
            self.data_association(imgs, bbs, bbs_norm, frame_num, bookkeeping)
            self.update_existing_tracks(bbs, bbs_norm, frame_num, bookkeeping)
            self.start_new_tracks(imgs, bbs, bbs_norm, frame_num, bookkeeping)

    def initialize_track_mems(self, imgs, bbs, bbs_norm):

        cur_detbb_num = np.shape(imgs)[0]
        assert(cur_detbb_num == np.shape(bbs)[0])

        mask_np = np.ones((cur_detbb_num, 1, 1))
        state_np = np.zeros((cur_detbb_num, 2 * self.model_info['mot_hidden_dim']))
        # det_num == trk_num, that's why I use cur_detbb_num repeatedly
        h_np, c_np = self.update_lstm_states(bbs_norm, cur_detbb_num, cur_detbb_num, mask_np, state_np)

        assert(np.shape(h_np)[0] == cur_detbb_num)
        assert(np.shape(c_np)[0] == cur_detbb_num)
        h_np, c_np = self.get_lstm_states_new(h_np, c_np, cur_detbb_num)

        return (h_np, c_np)

    def update_lstm_states(self, det_bbs_np, det_num, trk_num, valid_mot_data_np, motlstm_state_np):
        
        feed_dict = {
            self.tf_plh['detection_bboxes']: det_bbs_np,
            self.tf_plh['valid_mot_data']: valid_mot_data_np, 
            self.tf_plh['start_offset']: np.zeros(det_num, dtype=np.int32),
            self.tf_plh['end_offset']: np.zeros(det_num, dtype=np.int32),
            self.tf_plh['num_step_by_user']: 1,
            self.tf_plh['istate_mot']: motlstm_state_np,
            self.tf_plh['track_len']: np.ones(det_num, dtype=np.int32),
            self.tf_plh['c_mot_states_plh']: motlstm_state_np,
            self.tf_plh['mid_missdet_num']: np.zeros(det_num, dtype=np.int32),
            self.tf_plh['first_missdet_num']: np.zeros(det_num, dtype=np.int32),
            self.tf_plh['detbb_num']: det_num,
            self.tf_plh['track_num']: trk_num
        }

        tf_ops = [
            self.tf_ops['h_mot_states_test'],
            self.tf_ops['c_mot_states_last_test']
        ]

        h_np, c_np = self.sess.run(tf_ops, feed_dict=feed_dict)

        return (h_np, c_np)

    def data_association(self, imgs, bbs, bbs_norm, frame_num, bookkeeping):

        # cur_detbb_num = np.shape(imgs)[0]
        track_memory_sel, track_bb_sel, track_ids_sel = self.collect_track_memory()

        assert(np.shape(track_memory_sel)[0] == len(track_ids_sel))

        matching = []
        matching_score = []
        inds_sel1 = []
        track_i_sel = []   
        h_np = []
        c_np = []
        if imgs is not None and np.shape(track_memory_sel)[0] > 1:

            detbb_num = np.shape(imgs)[0]
            assert(detbb_num == np.shape(bbs)[0])
            track_num = np.shape(track_memory_sel)[0]

            softmax_np, h_np, c_np = self.compute_softmax_score(
                                         bbs_norm,
                                         bbs,
                                         track_bb_sel,
                                         detbb_num,
                                         track_num,
                                         track_memory_sel
                                     )
            
            matching, matching_score = self.solve_greedy_matching(
                                           softmax_np,
                                           track_memory_sel,
                                           track_num,
                                           detbb_num,
                                           track_ids_sel,
                                           bbs,
                                           frame_num
                                       )

            h_np, c_np, inds_sel1, track_i_sel = self.get_lstm_states_for_matched_tracks(
                                                     matching,
                                                     self.model_info['mot_hidden_dim'],
                                                     h_np,
                                                     c_np,
                                                     track_num,
                                                     detbb_num
                                                 )
            
        bookkeeping['matching'] = matching
        bookkeeping['matching_score'] = matching_score
        bookkeeping['track_memory_sel'] = track_memory_sel
        bookkeeping['track_ids_sel'] = track_ids_sel
        bookkeeping['inds_sel1'] = inds_sel1
        bookkeeping['track_i_sel'] = track_i_sel
        bookkeeping['h_np'] = h_np
        bookkeeping['c_np'] = c_np
    
    def collect_track_memory(self):

        all_ids = sorted(self.all_tracks.keys())

        sel_ids = []
        for i in range(0, len(all_ids)):
            total_num = self.all_tracks[all_ids[i]]['real_det_num'] + \
                        self.all_tracks[all_ids[i]]['miss_det_num']

            if self.all_tracks[all_ids[i]]['real_det_num'] > 0:
                if (((float(self.all_tracks[all_ids[i]]['real_det_num']) / total_num) >= self.det_ratio_thr) and
                    (self.all_tracks[all_ids[i]]['last_miss_det_num'] <= self.N_miss_max)):
                    sel_ids.append(all_ids[i])
            else:
                assert((float(self.all_tracks[all_ids[i]]['real_det_num']) / total_num) < 1.0)
        
        track_mem_sel = np.zeros((len(sel_ids), 2 * self.model_info['mot_hidden_dim']))
        track_bb_sel = np.zeros((len(sel_ids), self.cfg_train.MOT_INPUT_DIM))
        for i in range(0, len(sel_ids)):
            track_mem_sel[i, :] = self.all_tracks[sel_ids[i]]['c_states']
            track_bb_sel[i, :] = self.all_tracks[sel_ids[i]]['bb'][:self.cfg_train.MOT_INPUT_DIM]
            assert(np.shape(self.all_tracks[sel_ids[i]]['bb'])[0] == (self.cfg_train.MOT_INPUT_DIM + 1))
       
        return (track_mem_sel, track_bb_sel, sel_ids)

    def compute_softmax_score(self, bbs_norm, bbs, track_bbs, detbb_num, track_num, track_memory_sel):
                               
        feed_dict = {
            self.tf_plh['detection_bboxes']: bbs_norm,
            self.tf_plh['valid_mot_data']: np.ones((detbb_num, 1, 1)),
            self.tf_plh['start_offset']: np.zeros(detbb_num, dtype=np.int32),
            self.tf_plh['end_offset']: np.zeros(detbb_num, dtype=np.int32),
            self.tf_plh['num_step_by_user']: 1,
            self.tf_plh['istate_mot']: track_memory_sel,
            self.tf_plh['track_len']: np.ones(track_num, dtype=np.int32),
            self.tf_plh['c_mot_states_plh']: track_memory_sel,
            self.tf_plh['mid_missdet_num']: np.zeros(detbb_num, dtype=np.int32),
            self.tf_plh['first_missdet_num']: np.zeros(detbb_num, dtype=np.int32),
            self.tf_plh['det_bbox_org']: bbs[:, :self.cfg_train.MOT_INPUT_DIM],
            self.tf_plh['trk_bbox_org']: track_bbs,
            self.tf_plh['detbb_num']: detbb_num,
            self.tf_plh['track_num']: track_num,
            self.tf_plh['sel_indices']: np.ones((track_num, detbb_num, 1))
        }

        tf_ops = [
            self.tf_ops['softmax_out'],
            self.tf_ops['h_mot_states'],
            self.tf_ops['c_mot_states_last']
        ]

        # (N2 * N1) x 2
        softmax_np, h_np, c_np = self.sess.run(tf_ops, feed_dict=feed_dict)

        return (softmax_np, h_np, c_np)

    def update_existing_tracks(self, bbs, bbs_norm, frame_num, bookkeeping):                
        
        missing_det_tracks = {}
        track_i_all = range(len(bookkeeping['track_ids_sel']))
        track_i_left = np.setdiff1d(track_i_all, bookkeeping['track_i_sel'])

        if len(track_i_left) > 0:

            track_memory_left = np.zeros((len(track_i_left), 2 * self.model_info['mot_hidden_dim']))
            track_ids_left = np.zeros(len(track_i_left))
            for kkk in range(0, len(track_i_left)):
                track_i_cur = track_i_left[kkk]
                track_memory_left[kkk, :] = bookkeeping['track_memory_sel'][track_i_cur, :]
                track_ids_left[kkk] = bookkeeping['track_ids_sel'][track_i_cur]

            det_num = len(track_i_left)
            det_bbs_np = np.zeros((det_num, self.cfg_train.MOT_INPUT_DIM))
            valid_mot_data_np = np.zeros((det_num, 1, 1)) # use zero mask to make zero input
            motlstm_state_np = track_memory_left

            # det_num == trk_num, that's why det_num is used repeatedly
            h_np_left, c_np_left = self.update_lstm_states(
                                       det_bbs_np,
                                       det_num,
                                       det_num,
                                       valid_mot_data_np,
                                       motlstm_state_np
                                   )

            missing_det_tracks['track_ids'] = track_ids_left
            missing_det_tracks['track_index'] = track_i_left
            h_np_left, c_np_left = self.get_lstm_states_new(h_np_left, c_np_left, det_num)
            missing_det_tracks['c_states'] = c_np_left
            missing_det_tracks['h_states'] = h_np_left

        self.update_tracks(
            bookkeeping['h_np'],
            bookkeeping['c_np'],
            None,
            bbs,
            bbs_norm,
            bookkeeping['track_ids_sel'],
            bookkeeping['matching'],
            bookkeeping['matching_score'],
            frame_num,
            self.model_info['mot_hidden_dim'],
            network='mot_lstm',
            missdet_tracks=missing_det_tracks
        )
        
        self.delete_dummy_tracks(frame_num)

    def start_new_tracks(self, imgs, bbs, bbs_norm, frame_num, bookkeeping): 

        if imgs is not None:

            # start new tracks from detections which are not selected
            all_inds = range(np.shape(imgs)[0])
            inds_sel2 = np.setdiff1d(all_inds, bookkeeping['inds_sel1'])

            if len(inds_sel2) > 0:

                bbs_norm_sel2 = self.pick_dets(bbs_norm, inds_sel2)
                cur_detbb_num = np.shape(bbs_norm_sel2)[0]
                c_np = np.zeros((cur_detbb_num, 2 * self.model_info['mot_hidden_dim']))
                assert(cur_detbb_num == len(inds_sel2))

                det_bbs_np = bbs_norm_sel2
                det_num = cur_detbb_num
                valid_mot_data_np = np.ones((cur_detbb_num, 1, 1))
                motlstm_state_np = c_np   
                
                # det_num == trk_num, that's why det_num is used repeatedly
                h_np, c_np = self.update_lstm_states(
                                 det_bbs_np,
                                 det_num,
                                 det_num,
                                 valid_mot_data_np,
                                 motlstm_state_np
                             )

                assert(np.shape(h_np)[0] == cur_detbb_num)
                assert(np.shape(c_np)[0] == cur_detbb_num)
                h_np, c_np = self.get_lstm_states_new(h_np, c_np, cur_detbb_num)

                self.initialize_tracks(
                    h_np,
                    c_np,
                    None,
                    bbs,
                    bbs_norm,
                    inds_sel2,
                    frame_num,
                    self.model_info['mot_hidden_dim'],
                    is_dummy=False,
                    network='mot_lstm'
                )


class Greedy_Tracker_APP_MOT(Greedy_Tracker_APP_BLSTM, Greedy_Tracker_MOT_LSTM):

    def __init__(self, cfg_tracker, cfg_train, tf_ops, tf_placeholders, session):

        super(Greedy_Tracker_APP_MOT, self).__init__(cfg_tracker, cfg_train, tf_ops, tf_placeholders, session)

    def run(self, bbs, bbs_norm, imgs, frame_num):                

        # first frame
        if len(self.all_tracks.keys()) == 0 and imgs is not None:

            mem_np = self.initialize_track_mems(imgs, bbs, bbs_norm)            
            h_app_np, c_app_np, memory_app_np, h_mot_np, c_mot_np = mem_np

            cur_detbb_num = np.shape(imgs)[0]
            self.initialize_tracks(
                h_app_np, 
                c_app_np,
                memory_app_np,
                h_mot_np,
                c_mot_np,
                bbs,
                bbs_norm,
                np.array(range(cur_detbb_num)),
                frame_num,
                self.model_info['app_hidden_dim'],
                self.model_info['mot_hidden_dim'],
                is_dummy=False,
                network='app_mot_network'
            )            

        elif len(self.all_tracks.keys()) != 0:

            bookkeeping = {}
            self.data_association(imgs, bbs, bbs_norm, frame_num, bookkeeping)
            self.update_existing_tracks(bbs, bbs_norm, frame_num, bookkeeping)
            self.start_new_tracks(imgs, bbs, bbs_norm, frame_num, bookkeeping)

    def initialize_track_mems(self, imgs, bbs, bbs_norm):
        
        app_embed_np = self.precompute_app_features(imgs, bbs)

        cur_detbb_num = np.shape(imgs)[0]
        assert(cur_detbb_num == np.shape(bbs)[0])
        valid_mot_data = np.ones((cur_detbb_num, 1, 1))

        h_app_np, c_app_np, mem_app_np, h_mot_np, c_mot_np = self.update_lstm_states(
                                                                 app_embed_np,
                                                                 bbs_norm,
                                                                 valid_mot_data,
                                                                 cur_detbb_num
                                                             )    

        assert(np.shape(h_mot_np)[0] == cur_detbb_num)
        assert(np.shape(c_mot_np)[0] == cur_detbb_num)
        h_mot_np, c_mot_np = self.get_lstm_states_new(h_mot_np, c_mot_np, cur_detbb_num)

        mem_np = (h_app_np, c_app_np, mem_app_np, h_mot_np, c_mot_np)

        return mem_np

    def update_lstm_states(
            self,
            app_embed,
            bbs_norm,
            valid_mot_data,
            det_num
        ):

        c_app = np.zeros((det_num, 2 * self.model_info['app_hidden_dim']))
        c_mot = np.zeros((det_num, 2 * self.model_info['mot_hidden_dim']))

        feed_dict = {        
            self.tf_plh['detbb_num']: det_num,
            self.tf_plh['track_num']: det_num, # dummy
            self.tf_plh['app_embed_plh']: app_embed,
            self.tf_plh['istate_app']: c_app, 
            self.tf_plh['is_training']: False,
            self.tf_plh['num_step_by_user']: 1,
            self.tf_plh['frames_by_user']: np.array([1]),
            self.tf_plh['track_len']: np.ones(det_num, dtype=np.int32),
            self.tf_plh['track_len_offset']: np.zeros(det_num, dtype=np.int32),
            self.tf_plh['detection_bboxes']: bbs_norm,
            self.tf_plh['valid_mot_data']: valid_mot_data,
            self.tf_plh['start_offset']: np.zeros(det_num, dtype=np.int32),
            self.tf_plh['end_offset']: np.zeros(det_num, dtype=np.int32),
            self.tf_plh['istate_mot']: c_mot,                
            self.tf_plh['c_mot_states_plh']: c_mot,
            self.tf_plh['mid_missdet_num']: np.zeros(det_num, dtype=np.int32),
            self.tf_plh['first_missdet_num']: np.zeros(det_num, dtype=np.int32),
        }

        tf_ops = [
            self.tf_ops['h_app_states'],
            self.tf_ops['c_app_states'],
            self.tf_ops['h_app_track_memory'],
            self.tf_ops['h_mot_states_test'],
            self.tf_ops['c_mot_states_last_test']
        ]

        h_app_np, c_app_np, mem_app_np, h_mot_np, c_mot_np = self.sess.run(tf_ops, feed_dict=feed_dict)

        return (h_app_np, c_app_np, mem_app_np, h_mot_np, c_mot_np)
    
    def initialize_tracks(
            self,
            h_app,
            c_app,
            memory_app,
            h_mot,
            c_mot, 
            bbs,
            bbs_norm,
            det_ids,
            frame,
            app_hidden_dim,
            mot_hidden_dim,
            is_dummy,
            network
        ):

        h_app = np.reshape(h_app, (-1, app_hidden_dim))     
        assert(np.shape(memory_app)[0] == np.shape(h_app)[0])
        assert(np.shape(memory_app)[0] == np.shape(c_app)[0])
        assert(np.array_equal(h_app, c_app[:, app_hidden_dim:]))
        assert(np.shape(h_app)[0] == np.shape(c_app)[0])

        h_mot = np.reshape(h_mot, (-1, mot_hidden_dim))     
        assert(np.array_equal(h_mot, c_mot[:, mot_hidden_dim:]))
        assert(np.shape(h_mot)[0] == np.shape(c_mot)[0])
        assert(np.shape(h_mot)[0] == np.shape(h_app)[0])
        assert(np.shape(c_mot)[0] == np.shape(c_app)[0])

        if is_dummy == False:
            for i in range(0, np.shape(h_app)[0]):
                self.track_num += 1                
                # 1 x 1 x mem dim
                self.all_tracks[self.track_num]['mem_app'] = memory_app[i, :, :]
                # 1 x app state dim
                self.all_tracks[self.track_num]['h_app_states'] = h_app[i, :]
                # 1 x app state dim
                self.all_tracks[self.track_num]['c_app_states'] = c_app[i, :]
                # 1 x mot state dim
                self.all_tracks[self.track_num]['h_mot_states'] = h_mot[i, :]
                # 1 x mot state dim
                self.all_tracks[self.track_num]['c_mot_states'] = c_mot[i, :]
                self.all_tracks[self.track_num]['real_det_num'] = 1
                self.all_tracks[self.track_num]['miss_det_num'] = 0
                self.all_tracks[self.track_num]['last_miss_det_num'] = 0
                self.all_tracks[self.track_num]['bb'] = bbs[det_ids[i], :]
                self.all_tracks[self.track_num]['bb_norm'] = bbs_norm[det_ids[i], :]
                self.all_tracks[self.track_num]['frame'] = frame
                self.all_tracks[self.track_num]['th'] = [self.cls_thr]
                # self.all_tracks[self.track_num]['norm'] = np.linalg.norm(memory[i, :, :])

                self.result.append((frame, det_ids[i], 1.0, self.track_num))
        elif is_dummy == True:
            ct = -1
            for i in range(0, np.shape(memory_app)[0]):
                ct -= 1                
                # 1 x 1 x d
                self.all_tracks[ct]['mem_app'] = memory_app[i, :, :]
                # 1 x d
                self.all_tracks[ct]['h_app_states'] = h_app[i, :]
                # 1 x d
                self.all_tracks[ct]['c_app_states'] = c_app[i, :]
                # 1 x d
                self.all_tracks[ct]['h_mot_states'] = h_mot[i, :]
                # 1 x d
                self.all_tracks[ct]['c_mot_states'] = c_mot[i, :]
                self.all_tracks[ct]['real_det_num'] = 1
                self.all_tracks[ct]['miss_det_num'] = 0
                self.all_tracks[ct]['last_miss_det_num'] = 0
                self.all_tracks[ct]['bb'] = bbs[det_ids[i], :]
                self.all_tracks[ct]['bb_norm'] = bbs_norm[det_ids[i], :]
                self.all_tracks[ct]['frame'] = frame
                self.all_tracks[ct]['th'] = [self.cls_thr]
        else:
            raise NotImplementedError

    def data_association(self, imgs, bbs, bbs_norm, frame_num, bookkeeping):

        app_embed_np = []
        if imgs is not None:
            app_embed_np = self.precompute_app_features(imgs, bbs)            

        track_memory_sel, track_bb_sel, track_ids_sel = self.collect_track_memory()
        track_app_memory_sel, track_mot_memory_sel = track_memory_sel
        assert(np.shape(track_app_memory_sel)[0] == len(track_ids_sel))
        assert(np.shape(track_mot_memory_sel)[0] == len(track_ids_sel))

        matching = []
        matching_score = []
        inds_sel1 = []
        inds_sel1_tmp = []
        istates_sel = []
        track_i_sel = []   
        h_app_np = []
        c_app_np = []
        memory_app_np = []
        h_mot_np = []
        c_mot_np = []
        if imgs is not None and np.shape(track_app_memory_sel)[0] > 1:

            detbb_num = np.shape(imgs)[0]
            assert(detbb_num == np.shape(bbs)[0])
            track_num = np.shape(track_app_memory_sel)[0]
            indices = self.build_neg_mem_indices(track_num, detbb_num)

            softmax_np, m_app_states_np, h_mot_np, c_mot_np = self.compute_softmax_score(
                                                                  detbb_num,
                                                                  track_num,
                                                                  bbs_norm,
                                                                  app_embed_np,
                                                                  track_mot_memory_sel,
                                                                  track_app_memory_sel,
                                                                  indices
                                                              )

            matching, matching_score = self.solve_greedy_matching(
                                           softmax_np,
                                           m_app_states_np,
                                           track_num,
                                           detbb_num,
                                           track_ids_sel,
                                           bbs,
                                           frame_num
                                       )

            inds_sel1, istates_sel = self.pick_imgs_and_istates(
                                         track_ids_sel,
                                         matching,
                                         self.model_info['app_hidden_dim'],
                                     )

            h_mot_np, c_mot_np, inds_sel1_tmp, track_i_sel = self.get_lstm_states_for_matched_tracks(
                                                                 matching,
                                                                 self.model_info['mot_hidden_dim'],
                                                                 h_mot_np,
                                                                 c_mot_np,
                                                                 track_num,
                                                                 detbb_num
                                                             )                            

        assert(len(matching) == len(inds_sel1))
        assert(inds_sel1 == inds_sel1_tmp)

        bookkeeping['matching'] = matching
        bookkeeping['matching_score'] = matching_score
        bookkeeping['track_ids_sel'] = track_ids_sel        
        bookkeeping['inds_sel1'] = inds_sel1
        bookkeeping['istates_sel'] = istates_sel
        bookkeeping['track_i_sel'] = track_i_sel
        bookkeeping['h_app_np'] = h_app_np
        bookkeeping['c_app_np'] = c_app_np
        bookkeeping['memory_app_np'] = memory_app_np
        bookkeeping['h_mot_np'] = h_mot_np
        bookkeeping['c_mot_np'] = c_mot_np
        bookkeeping['app_embed_np'] = app_embed_np
        bookkeeping['track_mot_memory_sel'] = track_mot_memory_sel

    def collect_track_memory(self):

        all_ids = sorted(self.all_tracks.keys())

        sel_ids = []
        for i in range(0, len(all_ids)):
            total_num = self.all_tracks[all_ids[i]]['real_det_num'] + \
                        self.all_tracks[all_ids[i]]['miss_det_num']

            if self.all_tracks[all_ids[i]]['real_det_num'] > 0:
                if (((float(self.all_tracks[all_ids[i]]['real_det_num']) / total_num) >= self.det_ratio_thr) and
                    (self.all_tracks[all_ids[i]]['last_miss_det_num'] <= self.N_miss_max)):
                    sel_ids.append(all_ids[i])
            else:
                assert((float(self.all_tracks[all_ids[i]]['real_det_num']) / total_num) < 1.0)

        track_app_mem_sel = np.zeros((len(sel_ids), 1,  self.model_info['app_hidden_dim']))
        track_mot_mem_sel = np.zeros((len(sel_ids), 2 * self.model_info['mot_hidden_dim']))
        track_bb_sel = np.zeros((len(sel_ids), self.cfg_train.MOT_INPUT_DIM))
        for i in range(0, len(sel_ids)):
            track_app_mem_sel[i, :, :] = self.all_tracks[sel_ids[i]]['mem_app']
            track_mot_mem_sel[i, :] = self.all_tracks[sel_ids[i]]['c_mot_states']
            track_bb_sel[i, :] = self.all_tracks[sel_ids[i]]['bb'][:self.cfg_train.MOT_INPUT_DIM]
            assert(np.shape(self.all_tracks[sel_ids[i]]['bb'])[0] == (self.cfg_train.MOT_INPUT_DIM + 1))
        track_mem_sel = track_app_mem_sel, track_mot_mem_sel
     
        return (track_mem_sel, track_bb_sel, sel_ids)

    def compute_softmax_score(
            self,
            detbb_num,
            track_num,
            bbs_norm,
            app_embed,
            track_mot_memory_sel,
            track_app_memory_sel,
            indices,
        ):

        # istate_app serves as dummy variables here. 
        # It does not effect the results here
        feed_dict = {
            self.tf_plh['detbb_num']: detbb_num,
            self.tf_plh['track_num']: track_num,
            self.tf_plh['app_embed_plh']: app_embed,
            self.tf_plh['h_app_track_memory_plh']: track_app_memory_sel,
            self.tf_plh['is_training']: False,
            self.tf_plh['num_step_by_user']: 1,
            self.tf_plh['frames_by_user']: np.array([0]),
            self.tf_plh['indices_by_user']: indices,
            self.tf_plh['track_len']: np.ones(np.shape(app_embed)[0], dtype=np.int32),
            self.tf_plh['track_len_offset']: np.zeros(np.shape(app_embed)[0], dtype=np.int32),
            self.tf_plh['valid_app_data']: np.ones((detbb_num, 1, 1), dtype=np.int32),
            self.tf_plh['indices_for_mapping']: np.reshape(np.arange(detbb_num * 1, dtype=np.int32), (-1, 1)),
            self.tf_plh['image_batch_shape']: np.array([detbb_num * 1, self.cfg_train.APP_LAYER_DIM]),
            self.tf_plh['detection_bboxes']: bbs_norm,
            self.tf_plh['valid_mot_data']: np.ones((detbb_num, 1, 1)),
            self.tf_plh['start_offset']: np.zeros(detbb_num, dtype=np.int32),
            self.tf_plh['end_offset']: np.zeros(detbb_num, dtype=np.int32),             
            self.tf_plh['c_mot_states_plh']: track_mot_memory_sel,
            self.tf_plh['mid_missdet_num']: np.zeros(detbb_num, dtype=np.int32),
            self.tf_plh['first_missdet_num']: np.zeros(detbb_num, dtype=np.int32),
            self.tf_plh['app_frame_num']: np.zeros((track_num, 1))
        }

        tf_ops = [
            self.tf_ops['softmax_out'],
            self.tf_ops['m_states'],
            self.tf_ops['h_mot_states'], 
            self.tf_ops['c_mot_states_last']
        ]
            
        # (N2 * N1) x 2, (N2 * N1) x (2 * (d / feat_dim)) matrix
        softmax_np, m_app_states_np, h_mot_np, c_mot_np = self.sess.run(tf_ops, feed_dict=feed_dict)

        return (softmax_np, m_app_states_np, h_mot_np, c_mot_np)

    def pick_imgs_and_istates(self, track_ids, matching, hidden_dim):

        h_states_sel = np.zeros((len(matching), hidden_dim))
        c_states_sel = np.zeros((len(matching), 2 * hidden_dim))
        imgs_sel_inds = []
        for i in range(0, len(matching)):
            track_i = int(matching[i][0, 0])
            detbb_i = int(matching[i][0, 1])
            h_states_sel[i, :] = self.all_tracks[track_ids[track_i]]['h_app_states']
            c_states_sel[i, :] = self.all_tracks[track_ids[track_i]]['c_app_states']
            imgs_sel_inds.append(detbb_i)

        return (imgs_sel_inds, c_states_sel)

    def update_existing_tracks(self, bbs, bbs_norm, frame_num, bookkeeping):   

        h_app_np = bookkeeping['h_app_np']
        c_app_np = bookkeeping['c_app_np']
        memory_app_np = bookkeeping['memory_app_np']
        h_mot_np = bookkeeping['h_mot_np']
        c_mot_np = bookkeeping['c_mot_np']
        app_embed_np = bookkeeping['app_embed_np']

        track_i_all = range(len(bookkeeping['track_ids_sel']))
        track_i_left = np.setdiff1d(track_i_all, bookkeeping['track_i_sel'])
        track_num = len(track_i_left)
        missing_det_tracks = {}
        if track_num > 0:

            track_memory_left = np.zeros((track_num, 2 * self.model_info['mot_hidden_dim']))
            track_ids_left = np.zeros(track_num)
            for kkk in range(0, track_num):
                track_i_cur = track_i_left[kkk]
                track_memory_left[kkk, :] = bookkeeping['track_mot_memory_sel'][track_i_cur, :]
                track_ids_left[kkk] = bookkeeping['track_ids_sel'][track_i_cur]

            h_np_left, c_np_left = self.update_mot_lstm_state(track_num, track_memory_left) 

            missing_det_tracks['track_ids'] = track_ids_left
            missing_det_tracks['track_index'] = track_i_left
            h_np_left, c_np_left = self.get_lstm_states_new(h_np_left, c_np_left, track_num)
            missing_det_tracks['c_mot_states'] = c_np_left
            missing_det_tracks['h_mot_states'] = h_np_left

        if len(bookkeeping['inds_sel1']) > 0:              
                
            track_num = np.shape(bookkeeping['istates_sel'])[0]
            app_embed_sel_np = app_embed_np[bookkeeping['inds_sel1'], :]
            h_app_np, c_app_np, memory_app_np = self.update_app_lstm_state(
                                                    track_num,
                                                    app_embed_sel_np,
                                                    bookkeeping['istates_sel']
                                                )

        self.update_tracks(
            h_app_np,
            c_app_np,
            memory_app_np,
            h_mot_np,
            c_mot_np,
            bbs,
            bbs_norm,
            bookkeeping['track_ids_sel'],
            bookkeeping['matching'],
            bookkeeping['matching_score'],
            frame_num,
            self.model_info['app_hidden_dim'], 
            self.model_info['mot_hidden_dim'],
            network='app_mot_network',
            missdet_tracks=missing_det_tracks
        )

        self.delete_dummy_tracks(frame_num)

    def update_tracks(
            self,
            h_app,
            c_app,
            memory_app,
            h_mot,
            c_mot,
            bbs,
            bbs_norm,
            track_ids, 
            matching,
            matching_score,
            frame, 
            app_hidden_dim,
            mot_hidden_dim,
            network,
            missdet_tracks
        ):

        h_app = np.reshape(h_app, (-1, app_hidden_dim))
        if np.shape(c_app)[0] != 0:           
            assert(np.shape(memory_app)[0] == np.shape(h_app)[0])
            assert(np.shape(memory_app)[0] == np.shape(c_app)[0])
            assert(np.array_equal(h_app, c_app[:, app_hidden_dim:]))
            assert(np.shape(memory_app)[0] == np.shape(h_mot)[0])
            assert(np.shape(memory_app)[0] == np.shape(c_mot)[0])
        assert(len(matching) == len(matching_score))

        track_ids_sel1 = []
        for i in range(0, len(matching)):   
            track_i = int(matching[i][0, 0])
            detbb_i = int(matching[i][0, 1])            
            self.all_tracks[track_ids[track_i]]['mem_app'] = memory_app[i, :, :]
            self.all_tracks[track_ids[track_i]]['h_app_states'] = h_app[i, :]
            self.all_tracks[track_ids[track_i]]['c_app_states'] = c_app[i, :]
            self.all_tracks[track_ids[track_i]]['h_mot_states'] = h_mot[i, :]
            self.all_tracks[track_ids[track_i]]['c_mot_states'] = c_mot[i, :]       
            self.all_tracks[track_ids[track_i]]['real_det_num'] += 1
            self.all_tracks[track_ids[track_i]]['last_miss_det_num'] = 0
            self.all_tracks[track_ids[track_i]]['bb'] = bbs[detbb_i, :]
            self.all_tracks[track_ids[track_i]]['bb_norm'] = bbs_norm[detbb_i, :]
            self.all_tracks[track_ids[track_i]]['frame'] = frame
            self.all_tracks[track_ids[track_i]]['th'] = self.all_tracks[track_ids[track_i]]['th'] \
                                                        + [matching_score[i]]
            # self.all_tracks[track_ids[track_i]]['norm'] = np.linalg.norm(memory[i, :, :])
            self.result.append((frame, detbb_i, 1.0, track_ids[track_i]))
            track_ids_sel1.append(track_ids[track_i])
                     
        # update non matched tracks with dummy detections
        track_ids_sel2 = np.setdiff1d(track_ids, track_ids_sel1)

        if len(track_ids_sel2) > 0:
            assert(np.array_equal(track_ids_sel2, missdet_tracks['track_ids']))

        for i in range(0, len(track_ids_sel2)):
            # skip dummy track
            if track_ids_sel2[i] < 0:
                continue
            self.all_tracks[track_ids_sel2[i]]['miss_det_num'] += 1
            self.all_tracks[track_ids_sel2[i]]['last_miss_det_num'] += 1
            self.result.append((frame, None, None, track_ids_sel2[i]))
            
            self.all_tracks[track_ids_sel2[i]]['h_mot_states'] = missdet_tracks['h_mot_states'][i, :]
            self.all_tracks[track_ids_sel2[i]]['c_mot_states'] = missdet_tracks['c_mot_states'][i, :]
            assert(track_ids_sel2[i] == missdet_tracks['track_ids'][i])

    def start_new_tracks(self, imgs, bbs, bbs_norm, frame_num, bookkeeping):
         
        if imgs is not None:

            # start new tracks from detections which are not selected
            all_inds = range(np.shape(imgs)[0])
            inds_sel2 = np.setdiff1d(all_inds, bookkeeping['inds_sel1'])

            if len(inds_sel2) > 0:
                    
                bbs_norm_sel2 = self.pick_dets(bbs_norm, inds_sel2)

                cur_detbb_num = len(inds_sel2)
                assert(cur_detbb_num == len(inds_sel2))
                app_embed_sel2_np = bookkeeping['app_embed_np'][inds_sel2, :]
                valid_mot_data = np.ones((cur_detbb_num, 1, 1))

                h_app_np, c_app_np, memory_app_np, h_mot_np, c_mot_np = self.update_lstm_states(
                                                                            app_embed_sel2_np,
                                                                            bbs_norm_sel2,
                                                                            valid_mot_data,
                                                                            cur_detbb_num
                                                                        )            

                assert(np.shape(h_mot_np)[0] == cur_detbb_num)
                assert(np.shape(c_mot_np)[0] == cur_detbb_num)   
                h_mot_np, c_mot_np = self.get_lstm_states_new(h_mot_np, c_mot_np, cur_detbb_num)

                self.initialize_tracks(
                    h_app_np, 
                    c_app_np,
                    memory_app_np,
                    h_mot_np,
                    c_mot_np,
                    bbs,
                    bbs_norm,
                    inds_sel2,
                    frame_num,
                    self.model_info['app_hidden_dim'],
                    self.model_info['mot_hidden_dim'],
                    is_dummy=False,
                    network='app_mot_network'
                )                    
    
    def update_mot_lstm_state(self, track_num, track_memory_left):

        feed_dict = {
            self.tf_plh['detbb_num']: track_num, # dummy
            self.tf_plh['track_num']: track_num,
            self.tf_plh['detection_bboxes']: np.zeros((track_num, self.cfg_train.MOT_INPUT_DIM)),
            self.tf_plh['valid_mot_data']: np.zeros((track_num, 1, 1)), # use zero mask to make zero input
            self.tf_plh['start_offset']: np.zeros(track_num, dtype=np.int32),
            self.tf_plh['end_offset']: np.zeros(track_num, dtype=np.int32),
            self.tf_plh['num_step_by_user']: 1,
            self.tf_plh['istate_mot']: track_memory_left,
            self.tf_plh['track_len']: np.ones(track_num, dtype=np.int32),
            self.tf_plh['c_mot_states_plh']: track_memory_left,
            self.tf_plh['mid_missdet_num']: np.zeros(track_num, dtype=np.int32),
            self.tf_plh['first_missdet_num']: np.zeros(track_num, dtype=np.int32),
        }

        tf_ops = [
            self.tf_ops['h_mot_states_test'],
            self.tf_ops['c_mot_states_last_test']
        ]

        h_np_left, c_np_left = self.sess.run(tf_ops, feed_dict=feed_dict)     

        return (h_np_left, c_np_left)

    def update_app_lstm_state(self, track_num, app_embed_sel, istates_sel):

        # update tracks
        feed_dict = {
            self.tf_plh['detbb_num']: track_num, # dummy
            self.tf_plh['track_num']: track_num,
            self.tf_plh['app_embed_plh']: app_embed_sel,
            self.tf_plh['istate_app']: istates_sel,
            self.tf_plh['is_training']: False,
            self.tf_plh['num_step_by_user']: 1,
            self.tf_plh['frames_by_user']: np.array([1]),
            self.tf_plh['track_len']: np.ones(track_num, dtype=np.int32),
            self.tf_plh['track_len_offset']: np.zeros(track_num, dtype=np.int32),
        }

        tf_ops = [
            self.tf_ops['h_app_states'],
            self.tf_ops['c_app_states'],
            self.tf_ops['h_app_track_memory']
        ]

        h_app_np, c_app_np, memory_app_np = self.sess.run(tf_ops, feed_dict=feed_dict)

        return (h_app_np, c_app_np, memory_app_np)

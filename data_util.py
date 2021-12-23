import cv2
import os
import csv
import numpy as np
from six.moves import configparser
from sequence import get_initial_seq_tuple


# MOT Challenge dataset
class Data_MOT(object):

    def __init__(self, seq_name, data_path, type, visibility_thr):
        
        self.seq_name = seq_name
        self.orig_seq_name = seq_name
        self.seq_len = 0
        self.img_width = 0
        self.img_height = 0
        self.info_path = os.path.join(data_path, self.orig_seq_name, 'seqinfo.ini')
        self.gt_path = os.path.join(data_path, self.orig_seq_name, 'gt', 'gt.txt')
        self.img_path = os.path.join(data_path, self.orig_seq_name, 'img1')
        self.det_path = os.path.join(data_path, self.orig_seq_name, 'det_mapped')
        self.type = type
        self.data_in_frame = {}
        self.data_in_trk = {}
        self.data_missingdets_in_trk = {}
        self.dets_in_frame = {}
        self.dets_in_trk = {}
        self.dets_missingdets_in_trk = {}
        self.iou_det = {}
        self.iou_gt = {}
        self.dist_gt = {}
        self.gt_first_frame = float('inf')
        self.gt_last_frame = 0    
        self.det_first_frame = {}
        self.det_last_frame = {} 
        self.visibility_thr = visibility_thr
        self.data_for_verification_during_training = {}
        self.dets_for_verification_during_training = {}

        self.load_sequence_info()
        assert(self.img_width != 0 and self.img_height != 0)

        self.load_gt()

        if ('MOT17' in seq_name) or ('MOT15' in seq_name):
            self.load_det()

    def load_sequence_info(self):

        cfg_parser = configparser.ConfigParser()
        cfg_parser.read(self.info_path)

        self.img_width = int(cfg_parser.get('Sequence', 'imWidth'))
        self.img_height = int(cfg_parser.get('Sequence', 'imHeight'))
        self.seq_len = int(cfg_parser.get('Sequence', 'seqLength'))

    def load_images(self):

        img_files = os.listdir(self.img_path)
        img_files.sort()

        assert(self.seq_len == len(img_files))

        images_sz = (self.seq_len, self.img_height, self.img_width, 3)
        images = np.zeros(images_sz, dtype=np.uint8)
        for i in range(0, len(img_files)):
            img_path = os.path.join(self.img_path, img_files[i])
            img_tmp = cv2.imread(img_path)
            images[i, :, :, :] = img_tmp
            
            assert(img_tmp.shape[0] == self.img_height)
            assert(img_tmp.shape[1] == self.img_width)

        self.images = images

    def load_gt(self):
        """ 
        load gt tracks
        """

        last_frame = {}
        assert(os.path.isfile(self.gt_path) == True)
        with open(self.gt_path) as gt_file:
            reader = csv.reader(gt_file, delimiter=',')
            for data in reader:

                if ('MOT17' in self.seq_name) or ('MOT16' in self.seq_name):                    
                    if (int(data[7]) == 1 or int(data[7]) == 2 or int(data[7]) == 7) != True:
                        assert('MOT15' not in self.seq_name)
                        continue
                
                # ignore occluded bounding boxes
                visibility = float(data[8])
                assert(visibility >= 0 and visibility <= 1)
                if visibility < self.visibility_thr:
                    continue

                frame = float(data[0])
                track_id = float(data[1])
                # x, y need to be converted to 0-based from 1-based
                xmin = float(data[2]) - 1
                ymin = float(data[3]) - 1
                width = float(data[4])
                height = float(data[5])
                xmax = xmin + width
                ymax = ymin + height

                xmin = np.maximum(0.0, xmin)
                ymin = np.maximum(0.0, ymin)
                xmax = np.minimum(float(self.img_width), xmax)
                ymax = np.minimum(float(self.img_height), ymax)

                # remove some GT labels due to negative ymax and xmax values
                if track_id == 45 and self.seq_name == 'MOT17-10' and frame >= 366:
                    continue

                assert(xmax > 0 and ymax > 0)

                if track_id in last_frame.keys():
                    prev_frame = last_frame[track_id]
                    if frame - prev_frame != 1:
                        assert((frame - prev_frame) > 1)
                        # for data_in_frame
                        self.add_missing_detection_dif(self.data_in_frame, track_id, \
                                                       frame, prev_frame)
                        # for data_in_trk
                        self.add_missing_detection_dit(self.data_in_trk, track_id, \
                                                       frame, prev_frame)

                data1 = [ymin, xmin, ymax, xmax, track_id]
                data1 = np.reshape(data1, (1, len(data1)))
                frame_num = int(frame)
                try:
                    tmp = (self.data_in_frame[frame_num], data1)
                    self.data_in_frame[frame_num] = np.concatenate(tmp, axis=0)
                except KeyError:
                    self.data_in_frame[frame_num] = data1

                data2 = [ymin, xmin, ymax, xmax, frame]
                data2 = np.reshape(data2, (1, len(data2)))
                id_num = int(track_id)
                try:
                    tmp = (self.data_in_trk[id_num], data2)
                    self.data_in_trk[id_num] = np.concatenate(tmp, axis=0)
                except KeyError:
                    self.data_in_trk[id_num] = data2
                
                if id_num not in self.data_for_verification_during_training.keys():
                    self.data_for_verification_during_training[id_num] = {}
                assert(frame_num not in self.data_for_verification_during_training[id_num].keys())
                # same format (xmin, ymin, width, height) as in the "get_motlstm_orig_bb" function
                bbox_tmp = np.array([xmin, ymin, xmax - xmin, ymax - ymin])
                self.data_for_verification_during_training[id_num][frame_num] = bbox_tmp

                # update the last frame where there is an object detection
                last_frame[track_id] = frame

                # update first & last frame
                if frame_num < self.gt_first_frame:
                    self.gt_first_frame = frame_num

                if frame_num > self.gt_last_frame:
                    self.gt_last_frame = frame_num
       
        assert(self.gt_first_frame != float('inf') and self.gt_last_frame != 0)
        self.check_dif_and_dit(self.data_in_frame, self.data_in_trk)

    def load_det(self):
        """ 
        load noisy tracks constructed from object detections
        """

        if 'MOT17' in self.seq_name: 
            self.dets_in_frame['dpm'] = {}
            self.dets_in_frame['frcnn'] = {}
            self.dets_in_frame['sdp'] = {}
            self.dets_in_trk['dpm'] = {}
            self.dets_in_trk['frcnn'] = {}
            self.dets_in_trk['sdp'] = {}
            self.dets_for_verification_during_training['dpm'] = {}
            self.dets_for_verification_during_training['frcnn'] = {}
            self.dets_for_verification_during_training['sdp'] = {}
            self.det_first_frame['dpm'] = float('inf')
            self.det_first_frame['frcnn'] = float('inf')
            self.det_first_frame['sdp'] = float('inf')
            self.det_last_frame['dpm'] = 0
            self.det_last_frame['frcnn'] = 0
            self.det_last_frame['sdp'] = 0
            detectors = ['dpm', 'frcnn', 'sdp']
        elif 'MOT15' in self.seq_name:
            self.dets_in_frame['dpm'] = {}
            self.dets_in_trk['dpm'] = {}
            self.dets_for_verification_during_training['dpm'] = {}
            self.det_first_frame['dpm'] = float('inf')
            self.det_last_frame['dpm'] = 0
            detectors = ['dpm']
        else:
            raise NotImplementedError
        
        for detector in detectors:
            cur_det_path = os.path.join(self.det_path, detector, self.orig_seq_name + '.txt')
            last_frame = {}
            fp_cur_track_id = 0
            assert(os.path.isfile(cur_det_path) == True)
            with open(cur_det_path) as det_file:
                reader = csv.reader(det_file, delimiter=',')
                for data in reader:
                    frame = float(data[0])
                    track_id = float(data[1])
                    # x, y need to be converted to 0-based from 1-based
                    xmin = float(data[2]) - 1
                    ymin = float(data[3]) - 1
                    width = float(data[4])
                    height = float(data[5])
                    xmax = xmin + width
                    ymax = ymin + height

                    xmin = np.maximum(0.0, xmin)
                    ymin = np.maximum(0.0, ymin)
                    xmax = np.minimum(float(self.img_width), xmax)
                    ymax = np.minimum(float(self.img_height), ymax)

                    # # ID number for false positive detections is -1 
                    if track_id == -1:
                        continue

                    assert(xmax > 0 and ymax > 0)

                    if track_id in last_frame.keys():
                        prev_frame = last_frame[track_id]
                        if frame - prev_frame != 1:
                            assert((frame - prev_frame) > 1)
                            # for data_in_frame
                            self.add_missing_detection_dif(self.dets_in_frame[detector], track_id, \
                                                           frame, prev_frame)
                            # for data_in_trk
                            self.add_missing_detection_dit(self.dets_in_trk[detector], track_id, \
                                                           frame, prev_frame)

                    data1 = [ymin, xmin, ymax, xmax, track_id]
                    data1 = np.reshape(data1, (1, len(data1)))
                    frame_num = int(frame)
                    try:
                        tmp = (self.dets_in_frame[detector][frame_num], data1)
                        self.dets_in_frame[detector][frame_num] = np.concatenate(tmp, axis=0)
                    except KeyError:
                        self.dets_in_frame[detector][frame_num] = data1

                    data2 = [ymin, xmin, ymax, xmax, frame]
                    data2 = np.reshape(data2, (1, len(data2)))
                    id_num = int(track_id)
                    try:
                        tmp = (self.dets_in_trk[detector][id_num], data2)
                        self.dets_in_trk[detector][id_num] = np.concatenate(tmp, axis=0)
                    except KeyError:
                        self.dets_in_trk[detector][id_num] = data2

                    if id_num not in self.dets_for_verification_during_training[detector].keys():
                        self.dets_for_verification_during_training[detector][id_num] = {}
                    # print(self.dets_for_verification_during_training[id_num])
                    assert(frame_num not in self.dets_for_verification_during_training[detector][id_num].keys())
                    # same format (xmin, ymin, width, height) as in the "get_motlstm_orig_bb" function
                    bbox_tmp = np.array([xmin, ymin, xmax - xmin, ymax - ymin])
                    self.dets_for_verification_during_training[detector][id_num][frame_num] = bbox_tmp
                    
                    # update the last frame where there is an object detection
                    last_frame[track_id] = frame

                    # update first & last frame
                    if frame_num < self.det_first_frame[detector]:
                        self.det_first_frame[detector] = frame_num

                    if frame_num > self.det_first_frame[detector]:
                        self.det_last_frame[detector] = frame_num

            assert(self.det_first_frame[detector] != float('inf') and self.det_last_frame[detector] != 0)
            self.check_dif_and_dit(self.dets_in_frame[detector], self.dets_in_trk[detector])

    def add_missing_detection_dif(self, data_in_frame, track_id, frame, prev_frame):
        
        miss_num = int(frame - prev_frame - 1)
        for i in range(0, miss_num):
            data1 = [0.0, 0.0, 0.0, 0.0, track_id]
            data1 = np.reshape(data1, (1, len(data1)))
            miss_fr = int(prev_frame + (i + 1))
            try:
                tmp = (data_in_frame[miss_fr], data1)
                data_in_frame[miss_fr] = np.concatenate(tmp, axis=0)
            except KeyError:
                data_in_frame[miss_fr] = data1

    def add_missing_detection_dit(self, data_in_trk, track_id, frame, prev_frame):
        
        track_id = int(track_id)
        assert(data_in_trk[track_id][-1, -1] == prev_frame)
        miss_num = int(frame - prev_frame - 1)
        miss_data = np.zeros((miss_num, 5))
        for i in range(0, miss_data.shape[0]):
            miss_data[i, 4] = prev_frame + (i + 1)
        tmp = (data_in_trk[track_id], miss_data)
        data_in_trk[track_id] = np.concatenate(tmp, axis=0)

    def check_dif_and_dit(self, dpf, dpi):

        all_ids = dpi.keys()
        for cur_id in all_ids:
            dpi_sel = dpi[cur_id]
            # make sure the length of any track is greater than 1
            # except for false positive detections
            if dpi_sel.shape[0] < 2:
                assert(dpi_sel.shape[0] == 1)
                # assert(np.sum(dpi_sel[0, :4]) == 0)
                if np.sum(dpi_sel[0, :4]) != 0:
                    print(cur_id, 'length1', dpi_sel)
            assert(np.sum(np.diff(dpi_sel[:, -1])) == (dpi_sel.shape[0] - 1))
            for i in range(0, dpi_sel.shape[0]):
                frame_sel = dpi_sel[i, -1]
                dpf_sel = dpf[int(frame_sel)]
                ind_sel = dpf_sel[:, -1] == cur_id
                assert(np.sum(ind_sel) == 1)
                assert(np.array_equal(dpf_sel[ind_sel, :4].reshape(-1), dpi_sel[i, :4]))

    def get_roi_in_image(self, image_cache, bb, is_sequence_flipping):
        """
        return a cropped image for the input bounding box
        """

        ymin = bb['ymin']
        xmin = bb['xmin']
        ymax = bb['ymax']
        xmax = bb['xmax']
        frame = bb['frame']
        
        if frame not in image_cache.keys():                    
            img_path = os.path.join(self.img_path, ('%06d.jpg' % frame))
            image_cache[frame] = cv2.imread(img_path)
        img_tmp = image_cache[frame]

        assert(img_tmp.shape[0] == self.img_height)
        assert(img_tmp.shape[1] == self.img_width)

        image_sel = img_tmp[ymin:ymax, xmin:xmax, :]

        if is_sequence_flipping is True:
            image_sel = cv2.flip(image_sel, 1)

        return image_sel


# MOT Challenge public detection
class Detections_MOT(object):

    def __init__(self, seq_name, seq_num, cfg_tracker, cfg_train):
        self.seq_name = seq_name
        self.seq_num = seq_num
        self.seq_len = 0
        self.img_width = 0
        self.img_height = 0
        self.data_path = cfg_tracker.data_path[seq_name]
        self.info_path = cfg_tracker.info_path[seq_name]
        self.img_path = cfg_tracker.img_path[seq_name]        
        self.detections = {}
        self.norm_detections = {}
        self.overlap = cfg_tracker.overlap
        self.cfg_tracker = cfg_tracker
        self.cfg_train = cfg_train

        # set variables
        self.set_img_info()
        self.load_detection()

        if self.cfg_tracker.is_nms is True:
            self.nms(self.overlap)

    def set_img_info(self):
        cfg_parser = configparser.ConfigParser()
        cfg_parser.read(self.info_path)

        self.img_width = int(cfg_parser.get('Sequence', 'imWidth'))
        self.img_height = int(cfg_parser.get('Sequence', 'imHeight'))
        self.seq_len = int(cfg_parser.get('Sequence', 'seqLength'))

    def load_detection(self):
        with open(self.data_path) as det_file:
            reader = csv.reader(det_file, delimiter=',')

            for det_info in reader:
                frame_num = int(det_info[0])
                # x, y need to be converted to 0-based from 1-based
                xmin = float(det_info[2]) - 1
                ymin = float(det_info[3]) - 1
                width = float(det_info[4])
                height = float(det_info[5])
                xmax = xmin + width
                ymax = ymin + height
                conf = float(det_info[6])

                # false positive filtering based on confidence score
                if conf < self.cfg_tracker.det_score[self.seq_num]:
                    continue

                # DPM false positive filtering based on unrealistic detection size 
                if self.cfg_tracker.detector == 'DPM' and ('MOT17-03' in self.seq_name or 'MOT17-04' in self.seq_name):
                    if height > 400:
                        continue

                tmp1 = np.array([xmin, ymin, width, height, conf])
                try:
                    tmp2 = (self.detections[frame_num], tmp1)
                    self.detections[frame_num] = np.vstack(tmp2)
                except KeyError:
                    self.detections[frame_num] = np.reshape(tmp1,(1,5))

                ymin = ymin/self.img_height
                xmin = xmin/self.img_width
                ymax = ymax/self.img_height
                xmax = xmax/self.img_width
                height = height/self.img_height
                width = width/self.img_width

                tmp1 = self.cfg_train.CONST*np.array([ymin, xmin, height, width, conf])
                try:
                    tmp2 = (self.norm_detections[frame_num], tmp1)
                    self.norm_detections[frame_num] = np.vstack(tmp2)
                except KeyError:
                    self.norm_detections[frame_num] = np.reshape(tmp1,(1,5))

    def nms(self, overlap):
        """
        python implementation of the following MATLAB NMS implementation:
        https://github.com/quantombone/exemplarsvm/blob/master/internal/esvm_nms.m
        """

        for i in range(1, self.seq_len+1):
            
            if i not in self.detections.keys():
                continue

            xmin = self.detections[i][:,0]
            ymin = self.detections[i][:,1]
            xmax = xmin + self.detections[i][:,2]
            ymax = ymin + self.detections[i][:,3]
            score = self.detections[i][:,4]

            area = np.multiply(xmax-xmin+1, ymax-ymin+1)
            I = np.argsort(score)
            pick = score*0
            counter = 0

            while I.size != 0:
                last = I.size-1
                k = I[last]
                pick[counter] = k
                counter = counter + 1

                xx1 = np.maximum(xmin[k], xmin[I[0:-1]])
                yy1 = np.maximum(ymin[k], ymin[I[0:-1]])
                xx2 = np.minimum(xmax[k], xmax[I[0:-1]])
                yy2 = np.minimum(ymax[k], ymax[I[0:-1]])

                w = np.maximum(0.0, xx2-xx1+1)
                h = np.maximum(0.0, yy2-yy1+1)

                # o = np.divide(np.multiply(w,h), area[I[0:-1]])
                o = np.divide(np.multiply(w,h), area[k]) # This works better for suppressing overlapping DPM detections 

                ind_sel = [last] + np.where(o > overlap)[0].tolist()

                I = np.delete(I, ind_sel)
            
            pick = pick[:counter].astype(int) 
            self.detections[i] = self.detections[i][pick,:]
            self.norm_detections[i] = self.norm_detections[i][pick,:]


class Data_generator(object):

    def __init__(self, seq_list, dataset, network, cfg):

        self.seq_list = seq_list
        self.cur_seq_idx = 0
        self.seq_list_len = len(seq_list)
        self.dataset = dataset
        self.network = network
        self.cfg = cfg

        self.motion_network_list = [
            'motion_lstm',
        ]

    def initialize_batch(self, batch_info):

        # set data sizes
        img_sz = (batch_info['batch_num'], batch_info['num_step'], self.cfg.IMAGE_HEIGHT, self.cfg.IMAGE_WIDTH, 3)
        bbox_sz = (batch_info['batch_num'], batch_info['num_step'], self.cfg.MOT_INPUT_DIM)
        valid_bbox_sz = (batch_info['batch_num'], batch_info['num_step'], 1)
        frame_num_sz = (batch_info['batch_num'], batch_info['num_step'])
        
        # initialize mini-batch
        mini_batch = {
            'images': np.zeros(img_sz, dtype=np.uint8),
            'norm_bboxes': np.zeros(bbox_sz), # training_bbs (tf variable name)
            'norm_bboxes_prev': np.zeros(bbox_sz), # training_prev_bbs
            'orig_bboxes_synced_with_app': np.zeros(bbox_sz), # training_orig_bbs
            'orig_bboxes_synced_with_mot': np.zeros(bbox_sz), # training_orig_bbs
            'orig_noisy_bboxes_synced_with_app': np.zeros(bbox_sz),
            'orig_noisy_bboxes_synced_with_mot': np.zeros(bbox_sz),
            'orig_noisy_bboxes_prev_synced_with_mot': np.zeros(bbox_sz),
            'valid_app_data': np.zeros(valid_bbox_sz), # training_valid_data
            'valid_mot_data': np.zeros(valid_bbox_sz), # training_valid_motion_data
            'trk_len': np.zeros(batch_info['batch_num']), # training_trlen
            'start_offset': np.zeros(batch_info['batch_num']), # training_s_offset
            'end_offset': np.zeros(batch_info['batch_num']), # training_e_offset
            'miss_det_ct': np.zeros(batch_info['batch_num']), # training_miss_det_ct
            'mid_miss_det_ct': np.zeros(batch_info['batch_num']), # training_mid_miss_det_ct
            'aug_miss_det_ct': np.zeros(batch_info['batch_num']), # training_miss_det_aug_ct
            'first_miss_det_ct': np.zeros(batch_info['batch_num']), # training_first_miss_det_ct
            'is_started': np.zeros(batch_info['batch_num'], dtype=np.bool), # training_is_started
            'iou': np.zeros(batch_info['batch_num']), # training_iou
            'mot_frame_num': np.zeros(frame_num_sz), # training_mot_frame_num
            'app_frame_num': np.zeros(frame_num_sz), # training_app_frame_num
            'app_prev_frame_num': np.zeros(frame_num_sz), # training_app_frame_num
            'det_bbox_frame': np.zeros(batch_info['batch_num']),
            'trk_bbox_frame': np.zeros(batch_info['batch_num']),
            'lstm_state_frame': np.zeros(batch_info['batch_num']),
            'last_orig_bbox': np.zeros((batch_info['batch_num'], self.cfg.MOT_INPUT_DIM)),
        }

        return mini_batch

    def increment_seq_idx(self):

        self.cur_seq_idx += 1

    def reset_seq_idx(self):

        self.cur_seq_idx = 0

    def get_detection_noise(self):
        """
        Generate mean and std for the selected detector's localization noise. 
        """

        assert(len(self.cfg.DETECTOR_TYPE) == 3)

        ind_sel = int(np.random.random_integers(0, len(self.cfg.DETECTOR_TYPE) - 1, 1))

        # calculated from the MOT Challenge public detections
        noise = {}
        if self.cfg.DETECTOR_TYPE[ind_sel] == 'DPM':
            noise['detector'] = self.cfg.DETECTOR_TYPE[ind_sel]
            # x, y, w, h
            noise['mean'] = np.array([-0.025776, 0.019751, 0.040595, -0.020983])
            noise['cov'] = np.array([[0.015649, 0.001560, -0.014201, -0.002859],
                                    [0.001560, 0.004303, -0.002953, -0.005546],
                                    [-0.014201, -0.002953, 0.027595, 0.005563],
                                    [-0.002859, -0.005546, 0.005563, 0.015359]])
        elif self.cfg.DETECTOR_TYPE[ind_sel] == 'FRCNN':
            noise['detector'] = self.cfg.DETECTOR_TYPE[ind_sel]
            noise['mean'] = np.array([-0.005104, -0.009417, -0.026622, 0.004360])
            noise['cov'] = np.array([[0.004299, 0.000177, -0.002897, -0.000583],
                                    [0.000177, 0.002352, -0.000236, -0.002378],
                                    [-0.002897, -0.000236, 0.006258, 0.001288],
                                    [-0.000583, -0.002378, 0.001288, 0.005804]])
        elif self.cfg.DETECTOR_TYPE[ind_sel] == 'SDP':
            noise['detector'] = self.cfg.DETECTOR_TYPE[ind_sel]
            noise['mean'] = np.array([0.001145, -0.014884, -0.016546, 0.016892])
            noise['cov'] = np.array([[0.008461, 0.000411, -0.005253, -0.000914],
                                    [0.000411, 0.005436, -0.000528, -0.005186],
                                    [-0.005253, -0.000528, 0.010813, 0.001749],
                                    [-0.000914, -0.005186, 0.001749, 0.008984]])
        else:
            raise NotImplementedError

        return noise

    def get_bbox(self, data_in_trk, trk_id, ind_sel):

        ymin, xmin, ymax, xmax, frame = data_in_trk[trk_id][ind_sel, :]
        bbox = {
            'ymin': ymin,
            'xmin': xmin,
            'ymax': ymax,
            'xmax': xmax,
            'frame': frame
        }

        return bbox

    def get_motion_lstm_input(self, bb, img_sz, is_sequence_flipping):

        img_height, img_width = img_sz

        ymin_norm = float(bb['ymin']) / img_height
        xmin_norm = float(bb['xmin']) / img_width
        height_norm = float(bb['ymax'] - bb['ymin']) / img_height
        width_norm = float(bb['xmax'] - bb['xmin']) / img_width

        if is_sequence_flipping is True:
            xmin_norm = float(img_width - bb['xmax'])/img_width

        assert(self.cfg.MOT_INPUT_DIM == 4)
        data_tmp = [ymin_norm, xmin_norm, height_norm, width_norm]

        return self.cfg.CONST*np.array(data_tmp)

    def get_bbox_orig(self, bb):

        height = bb['ymax'] - bb['ymin']
        width = bb['xmax'] - bb['xmin']
        bbox_org = np.array([bb['xmin'], bb['ymin'], width, height])

        return bbox_org

    def add_bb_noise(self, bb, img_sz, noise_sel, det_type):

        ymin = bb['ymin']
        xmin = bb['xmin']
        ymax = bb['ymax']
        xmax = bb['xmax']
        frame = bb['frame']

        img_height, img_width = img_sz

        w = xmax - xmin
        h = ymax - ymin

        std = self.cfg.JITTER_STD
        cond = True
        while cond:
            if det_type == 'det':
                ymin_new, xmin_new, h_new, w_new = jitter_bb(ymin, xmin, h, w, std)
            elif det_type == 'gt':
                ymin_new, xmin_new, h_new, w_new = jitter_bb_with_matched_noise(ymin, xmin, h, w, noise_sel, self.cfg)
                # ymin_new, xmin_new, h_new, w_new = jitter_bb(ymin, xmin, h, w, std)
            else:
                raise NotImplementedError

            ymax_new = ymin_new + h_new
            xmax_new = xmin_new + w_new

            xmin_new = np.maximum(xmin_new, 0.0)
            ymin_new = np.maximum(ymin_new, 0.0)
            xmax_new = np.minimum(xmax_new, float(img_width))
            ymax_new = np.minimum(ymax_new, float(img_height))

            # check to see if the new coordinates look reasonable
            if int(ymax_new) > int(ymin_new) and int(xmax_new) > int(xmin_new):                   
                IOU_tmp = calculate_IoU((ymin, xmin, ymax, xmax), (ymin_new, xmin_new, ymax_new, xmax_new))
                if IOU_tmp < 0.5:            
                    cond = True
                    print('iou is too low. resampling bounding box. %f' % IOU_tmp)
                else:
                    cond = False

        ymin_new = int(ymin_new)
        xmin_new = int(xmin_new)
        ymax_new = int(ymax_new)
        xmax_new = int(xmax_new)

        bb_new = {
            'ymin': ymin_new,
            'xmin': xmin_new,
            'ymax': ymax_new,
            'xmax': xmax_new,
            'frame': frame
        }

        return (bb_new, IOU_tmp)

    def is_missing_detection(self, bbox):

        return (bbox['ymin'] == 0 and bbox['xmin'] == 0 and bbox['ymax'] == 0 and bbox['xmax'] == 0)


class Random_tracking_episode(Data_generator):

    def __init__(self, seq_list, dataset, network, cfg):

        super(Random_tracking_episode, self).__init__(seq_list, dataset, network, cfg)

    def generate_tracking_episode(self):

        while True:
            # set a flag that decides whether or not to do sequence flipping augmentation
            is_sequence_flipping = bool(np.random.rand(1) > 0.5)
            noise_sel = self.get_detection_noise()

            seq_name, track_id = self.seq_list[self.cur_seq_idx]
            seq_obj = self.dataset[seq_name]

            self.increment_seq_idx()
            if self.cur_seq_idx == self.seq_list_len:
                # go back to the first sequence
                self.reset_seq_idx()

            batch_info, is_valid = self.get_tracks(seq_obj, track_id)
            # resample tracks in case there is no valid track in the selected frames
            if is_valid == False:
                continue

            mini_batch = self.initialize_batch(batch_info)

            image_cache = {}
            for i in range(0, batch_info['ids_sel'].shape[0]):

                track_id_sel = batch_info['ids_sel'][i]
                trk_data_sel = seq_obj.data_in_trk[track_id_sel]
                trk_info = self.get_track_sync_info(batch_info, trk_data_sel, i)
               
                iou_tmp = 0.0
                bbox_prev = {}
                bbox_prev['norm_bbox'] = None
                bbox_prev['noisy_orig_bbox'] = None
                bbox_prev['frame'] = None
                prev_frame = -1
                real_det_ct = 0
                first_det_appeared = False
                is_missing_detection_aug = bool(np.random.rand(1) > (1 - self.cfg.MISSING_AUG_RATE)) 
                miss_det_rate = np.random.uniform(low=self.cfg.MISSING_DET_RATE_MIN, high=self.cfg.MISSING_DET_RATE_MAX, size=1)
                for j in range(0, trk_info['trk_len'] - trk_info['offset']):

                    if j != 0 and j != (trk_info['trk_len'] - trk_info['offset'] -1) \
                       and is_missing_detection_aug == True \
                       and (bool(np.random.rand(1) > (1 - miss_det_rate)) == True) \
                       and first_det_appeared ==  True:
                        assert(prev_frame != -1)
                        mini_batch['mid_miss_det_ct'][i] += 1                    
                        mini_batch['mot_frame_num'][i, j - int(mini_batch['first_miss_det_ct'][i])] = (prev_frame + 1)
                        prev_frame += 1
                    else:
                        ind_sel = j + trk_info['start_idx'] + trk_info['offset']
                        bbox = self.get_bbox(seq_obj.data_in_trk, track_id_sel, ind_sel)

                        if j == 0: assert(bbox['frame'] == trk_info['start_frame'] + trk_info['offset'])

                        prev_frame = bbox['frame']

                        # missing detection
                        if self.is_missing_detection(bbox) == True  and first_det_appeared == False:
                            mini_batch['first_miss_det_ct'][i] += 1
                            continue
                        elif self.is_missing_detection(bbox) == True  and first_det_appeared == True:
                            mini_batch['mid_miss_det_ct'][i] += 1
                            mini_batch['mot_frame_num'][i, j - int(mini_batch['first_miss_det_ct'][i])] = bbox['frame']
                            continue
                        
                        real_det_ct += 1
                        first_det_appeared = True
                        bbox_new, iou_tmp = self.add_bb_noise(bbox, batch_info['img_sz'], noise_sel, 'gt')
                        
                        self.update_tracks_in_batch(
                            (i, j),
                            mini_batch,
                            batch_info,
                            seq_obj,
                            image_cache,
                            bbox,
                            bbox_new,
                            bbox_prev,
                            is_sequence_flipping
                        )

                        if real_det_ct == 1: assert((j - int(mini_batch['first_miss_det_ct'][i])) == 0)

                        if j == (trk_info['trk_len'] - trk_info['offset'] - 1): 
                            assert(j == (trk_info['end_idx'] - trk_info['start_idx'] - trk_info['offset']))                

                self.update_other_info_in_batch(i, mini_batch, batch_info, trk_info, real_det_ct, bbox['frame'], iou_tmp)
                        
            break    

        return self.preprocess_track_data(mini_batch, batch_info)

    def preprocess_track_data(self, mini_batch, batch_info):

        # prepare input to the network in the right format
        images_sz = (batch_info['batch_num'] * batch_info['num_step'], self.cfg.IMAGE_HEIGHT, self.cfg.IMAGE_WIDTH, 3)
        mini_batch['images'] = mini_batch['images'].astype(np.float32)
        mini_batch['images'] = np.reshape(mini_batch['images'], images_sz)
        mini_batch['norm_bboxes'] = np.reshape(mini_batch['norm_bboxes'], (-1, self.cfg.MOT_INPUT_DIM))
        mini_batch['norm_bboxes_prev'] = np.reshape(mini_batch['norm_bboxes_prev'], (-1, self.cfg.MOT_INPUT_DIM))
        mini_batch['orig_bboxes_synced_with_app'] = np.reshape(mini_batch['orig_bboxes_synced_with_app'], (-1, self.cfg.MOT_INPUT_DIM))
        mini_batch['orig_bboxes_synced_with_mot'] = np.reshape(mini_batch['orig_bboxes_synced_with_mot'], (-1, self.cfg.MOT_INPUT_DIM))
        mini_batch['orig_noisy_bboxes_synced_with_app'] = np.reshape(mini_batch['orig_noisy_bboxes_synced_with_app'], (-1, self.cfg.MOT_INPUT_DIM))
        mini_batch['orig_noisy_bboxes_synced_with_mot'] = np.reshape(mini_batch['orig_noisy_bboxes_synced_with_mot'], (-1, self.cfg.MOT_INPUT_DIM))
        mini_batch['orig_noisy_bboxes_prev_synced_with_mot'] = np.reshape(mini_batch['orig_noisy_bboxes_prev_synced_with_mot'], (-1, self.cfg.MOT_INPUT_DIM))

        # data centering and normalization
        mini_batch['images'] -= np.array(self.cfg.IMAGENET_MEAN)
        mini_batch['norm_bboxes'] -= np.array(self.cfg.MOTION_MEAN)
        mini_batch['norm_bboxes'] /= np.array(self.cfg.MOTION_STD)
        mini_batch['norm_bboxes_prev'] -= np.array(self.cfg.MOTION_MEAN)
        mini_batch['norm_bboxes_prev'] /= np.array(self.cfg.MOTION_STD)

        # convert from BGR to RGB
        mini_batch['images'] = mini_batch['images'][:, :, :, [2, 1, 0]]

        # provide frames that will be used for training (currently, I only use the last frame)
        frame_ind_sel = np.array([batch_info['num_step'] - 1])
        sample_frame_num = 1

        # provide indices for building memory that contains negative target information
        neg_mem_indices = build_neg_mem_indices(batch_info['batch_num'], batch_info['batch_num'])

        # generate a mask for ignoring training examples in which an invalid track or an invalid detection is selected
        trk_st_ind = batch_info['num_step'] - mini_batch['trk_len']
        sel_indices = self.get_sel_indices(batch_info, sample_frame_num, frame_ind_sel, trk_st_ind)

        np_indices = np.reshape(np.arange(batch_info['batch_num'] * batch_info['num_step'], dtype=np.int32), (-1, 1))
        app_shape = np.array([batch_info['batch_num'] * batch_info['num_step'], self.cfg.APP_LAYER_DIM], dtype=np.int32)
        mot_shape = np.array([batch_info['batch_num'] * batch_info['num_step'], self.cfg.MOT_INPUT_DIM], dtype=np.int32)

        training_data = {
            'images': mini_batch['images'],
            'norm_bboxes': mini_batch['norm_bboxes'], # boundingboxes
            'norm_bboxes_prev': mini_batch['norm_bboxes_prev'], # boundingboxes_prev
            'valid_app_data': mini_batch['valid_app_data'], # valid_data
            'valid_mot_data': mini_batch['valid_mot_data'], # motion_valid_data
            'start_offset': mini_batch['start_offset'], # s_offset
            'end_offset': mini_batch['end_offset'], # e_offset
            'end_offset_binary': np.zeros(batch_info['batch_num'], dtype=np.int32), # e_offset_binary
            'mid_missdet_num': mini_batch['mid_miss_det_ct'],
            'first_missdet_num': mini_batch['first_miss_det_ct'],
            'aug_miss_det_num': mini_batch['aug_miss_det_ct'], # miss_det_aug_num
            'is_started': mini_batch['is_started'],
            'track_len': mini_batch['trk_len'],
            'mot_frame_num': mini_batch['mot_frame_num'],
            'iou': mini_batch['iou'],
            'app_frame_num': mini_batch['app_frame_num'],
            'app_prev_frame_num': mini_batch['app_prev_frame_num'],
            'mot_frame_num': mini_batch['mot_frame_num'],
            'trk_bbox_frame': mini_batch['trk_bbox_frame'],
            'det_bbox_frame': mini_batch['det_bbox_frame'],
            'orig_bboxes_synced_with_app': mini_batch['orig_bboxes_synced_with_app'],
            'orig_bboxes_synced_with_mot': mini_batch['orig_bboxes_synced_with_mot'],
            'orig_noisy_bboxes_synced_with_app': mini_batch['orig_noisy_bboxes_synced_with_app'],
            'orig_noisy_bboxes_synced_with_mot': mini_batch['orig_noisy_bboxes_synced_with_mot'],
            'orig_noisy_bboxes_prev_synced_with_mot': mini_batch['orig_noisy_bboxes_prev_synced_with_mot'],
            'lstm_state_frame': mini_batch['lstm_state_frame'],
            'num_step': batch_info['num_step'],
            'batch_num': batch_info['batch_num'],
            'frame_ind_sel': frame_ind_sel, # sel_frames
            'neg_mem_indices':  neg_mem_indices, # neg_indices
            'sel_indices': sel_indices, # ignore_indices
            'mapping_indices': np_indices,
            'image_batch_shape': app_shape,
            'bbox_batch_shape': mot_shape # mot_shape
        }

        return training_data

    def get_sequence_len_for_diverse_sample(self):

        # randomly pick the length of the training sequence
        num_step = np.random.choice(range(self.cfg.MIN_STEP, self.cfg.MAX_STEP + 1), 1)
        num_step = int(num_step)

        return num_step

    def get_sel_indices(self, batch_info, sample_num, frames_sel_ind, trk_st_ind):

        track_num = batch_info['batch_num']
        detbb_num = batch_info['batch_num']
        
        assert(sample_num == 1)

        sel_inds = np.ones((track_num, detbb_num, sample_num))
        for i in range(track_num):
            for j in range(detbb_num):
                if (trk_st_ind[i] > (frames_sel_ind - 1)) or (trk_st_ind[j] > frames_sel_ind):
                    sel_inds[i, j, 0] = 0.0
                # Since I am using only the detections in the last frame for generating matching candidates for now,
                # trk_st_ind cannot be greater than frame_sel[k] 
                assert(trk_st_ind[j] <= frames_sel_ind)

        return sel_inds

    def get_tracks(self, seq_obj, track_id):

        num_step = self.get_sequence_len_for_diverse_sample()
        track_len = seq_obj.data_in_trk[track_id].shape[0]

        # when false positive detections are used during training,
        # they are not used as tracks because their length is 1 (shorter than MIN_STEP).
        # However, they can still be used as new detections in the selected end frame
        # (i.e. selected by the "find_all_track_candidates_from_end_frame" function)
        if track_len < num_step:
            if track_len >= self.cfg.MIN_STEP:
                num_step = track_len
            # skip the track if it's too short
            else:
                return ({}, False)

        # randomly choose the video frames to generate training data
        st_ind = np.random.randint(track_len - num_step + 1)
        ed_ind = st_ind + (num_step - 1)
        st_fr = seq_obj.data_in_trk[track_id][st_ind, -1]
        ed_fr = seq_obj.data_in_trk[track_id][ed_ind, -1]
        assert((ed_ind - st_ind + 1) == num_step)
        assert((ed_fr - st_fr + 1) == num_step)

        # find all tracks in the selected end frame and randomly pick some of them
        all_ids = self.find_all_track_candidates_from_end_frame(ed_fr, seq_obj.data_in_frame)

        if np.sum(all_ids == track_id) == 0:
            return ({}, False)

        batch_num = np.minimum(self.cfg.BATCH_NUM, all_ids.shape[0])
        # remove the track selected from the iterator because it's already included 
        all_ids = np.delete(all_ids, np.where(all_ids == track_id)[0])
        if np.shape(all_ids)[0] == 0:
            print("only positive example in a batch")
            return ({}, False)
        selected_ids = np.random.choice(all_ids, size=(batch_num - 1), replace=False)
        # always put the track selected from the iterator at the first index
        selected_ids = np.insert(selected_ids, 0, track_id)

        assert(selected_ids[0] == track_id)
        assert(np.sum(selected_ids == track_id) == 1)
        assert(np.shape(selected_ids)[0] > 1)
        assert(np.shape(selected_ids)[0] == batch_num)

        batch_info = {
            'batch_num': batch_num,
            'num_step': num_step,
            'ids_sel': selected_ids,
            'start_frame': st_fr,
            'end_frame': ed_fr,
            'img_sz': (seq_obj.img_height, seq_obj.img_width)
        }

        return (batch_info, True)

    def get_track_sync_info(self, batch_info, trk_data_sel, cur_batch_idx):

        # find indices for selected frames
        end_fr = trk_data_sel[-1, -1]
        start_fr = trk_data_sel[0, -1]
        if end_fr == start_fr:
            assert(trk_data_sel.shape[0] == 1)

        start_fr = np.maximum(batch_info['start_frame'], start_fr)
        trk_num_step = int(batch_info['end_frame'] - start_fr + 1)

        st_i = int(trk_data_sel.shape[0] - (end_fr - start_fr + 1))
        ed_i = int(st_i + (trk_num_step - 1))

        assert(start_fr == trk_data_sel[st_i, -1])
        assert(batch_info['end_frame'] == trk_data_sel[ed_i, -1])

        if batch_info['start_frame'] == start_fr:
            assert(trk_num_step == batch_info['num_step'])
        elif batch_info['start_frame'] < start_fr:
            assert(trk_num_step < batch_info['num_step'])
        else:
            raise ValueError

        # Note that the same axis 1 index in the training data containers doesn't mean 
        # that the data come from the same video frame.
        trk_ind_step = int(ed_i - st_i + 1)       
        assert(trk_ind_step == trk_num_step)     
        if cur_batch_idx == 0:
            # first track keeps the original length
            offset = 0
        else:
            # randomly select the start frame for each track within the selected frames
            offset = int(np.random.choice(range(0, (trk_ind_step - 1) + 1), 1))
        
        track_sync_info = {
            'start_idx': st_i,
            'end_idx': ed_i,
            'trk_len': trk_ind_step,
            'offset': offset,
            'start_frame': start_fr,
            'end_frame': end_fr   
        }

        return track_sync_info

    def update_tracks_in_batch(
            self,
            inds,
            mini_batch,
            batch_info,
            seq_obj,
            image_cache,
            bbox,
            bbox_new,
            bbox_prev,
            is_seq_flip
        ):

        i, j = inds

        miss_det_offset = int(mini_batch['mid_miss_det_ct'][i] + mini_batch['first_miss_det_ct'][i])
        app_time_idx = j - miss_det_offset
        mot_time_idx = j - int(mini_batch['first_miss_det_ct'][i])
        if self.network not in self.motion_network_list:
            roi_img = seq_obj.get_roi_in_image(image_cache, bbox_new, is_seq_flip)
            cnn_input_size = (self.cfg.IMAGE_WIDTH, self.cfg.IMAGE_HEIGHT)
            mini_batch['images'][i, app_time_idx, :, :, :] = cv2.resize(roi_img, cnn_input_size)
        else:
            assert(self.network in self.motion_network_list)

        if j == 0: assert(mini_batch['mid_miss_det_ct'][i] == 0)
        
        norm_bbox = self.get_motion_lstm_input(bbox_new, batch_info['img_sz'], is_seq_flip)
        orig_bbox = self.get_bbox_orig(bbox)
        noisy_orig_bbox = self.get_bbox_orig(bbox_new)

        if bbox_prev['frame'] != None:
            assert(bbox_prev['frame'] < batch_info['end_frame'])
            mini_batch['trk_bbox_frame'][i] = bbox_prev['frame']

        if bbox_prev['norm_bbox'] is None: bbox_prev['norm_bbox'] = norm_bbox
        if bbox_prev['noisy_orig_bbox'] is None: bbox_prev['noisy_orig_bbox'] = noisy_orig_bbox
        if bbox_prev['frame'] is None: bbox_prev['frame'] = bbox['frame']

        mini_batch['norm_bboxes'][i, mot_time_idx, :] = norm_bbox
        # "mot_time_idx:" used intead of "mot_time_idx" index here to prevent the last element from being zeros
        mini_batch['norm_bboxes_prev'][i, mot_time_idx:, :] = bbox_prev['norm_bbox']
        # store this info to use in gating during training
        mini_batch['orig_bboxes_synced_with_app'][i, app_time_idx, :] = orig_bbox
        mini_batch['orig_bboxes_synced_with_mot'][i, mot_time_idx, :] = orig_bbox
        mini_batch['orig_noisy_bboxes_synced_with_app'][i, app_time_idx, :] = noisy_orig_bbox
        mini_batch['orig_noisy_bboxes_synced_with_mot'][i, mot_time_idx, :] = noisy_orig_bbox
        mini_batch['orig_noisy_bboxes_prev_synced_with_mot'][i, mot_time_idx:, :] = bbox_prev['noisy_orig_bbox']
        mini_batch['valid_app_data'][i, app_time_idx, 0] = 1.0
        mini_batch['valid_mot_data'][i, mot_time_idx, 0] = 1.0
        mini_batch['app_frame_num'][i, app_time_idx] = bbox['frame']
        mini_batch['app_prev_frame_num'][i, app_time_idx:] = bbox_prev['frame']
        mini_batch['mot_frame_num'][i, mot_time_idx] = bbox['frame']
        bbox_prev['norm_bbox'] = norm_bbox
        bbox_prev['noisy_orig_bbox'] = noisy_orig_bbox
        bbox_prev['frame'] = bbox['frame']

    def update_other_info_in_batch(self, i, mini_batch, batch_info, trk_info, real_det_ct, frame, iou_tmp):

        end_ind = int(trk_info['trk_len']  - trk_info['offset'] - mini_batch['first_miss_det_ct'][i])
        assert(end_ind > 0)
        assert(np.sum(np.diff(mini_batch['mot_frame_num'][i, :end_ind])) == \
            (np.shape(mini_batch['mot_frame_num'][i, :end_ind])[0] - 1))
        # if np.sum(training_mot_frame_num[i, end_ind:]) != 0:
        #     print(training_mot_frame_num[i, :])
        #     print(training_mot_frame_num[i, :])
        #     print(training_mot_frame_num[i, end_ind:])
        #     print(training_mot_frame_num[i, end_ind:])
        # assert(np.sum(training_mot_frame_num[i, end_ind:]) == 0)

        # last frame must be equal to the end frame that was selected earlier
        assert(frame == batch_info['end_frame'])

        mini_batch['iou'][i] = iou_tmp
        mini_batch['trk_len'][i] = trk_info['trk_len']  - trk_info['offset'] - mini_batch['mid_miss_det_ct'][i] \
                            - mini_batch['first_miss_det_ct'][i]
        mini_batch['start_offset'][i] = trk_info['offset'] + (batch_info['num_step'] - trk_info['trk_len'] ) + mini_batch['first_miss_det_ct'][i]
        mini_batch['end_offset'][i] = 0
        mini_batch['det_bbox_frame'][i] = batch_info['end_frame']

        assert(real_det_ct == mini_batch['trk_len'][i])
        assert(batch_info['num_step'] == mini_batch['trk_len'][i] + mini_batch['start_offset'][i] \
                        + mini_batch['end_offset'][i] + mini_batch['mid_miss_det_ct'][i])
        assert(trk_info['trk_len'] == mini_batch['trk_len'][i] + mini_batch['start_offset'][i] \
                            + mini_batch['end_offset'][i] + mini_batch['mid_miss_det_ct'][i] \
                            - (batch_info['num_step'] - trk_info['trk_len']))
        
        assert(mini_batch['trk_len'][i] != 0)
        if mini_batch['trk_len'][i] == 1: assert(mini_batch['start_offset'][i] == (batch_info['num_step'] - 1))

    def find_all_track_candidates_from_sel_frames(self, start_frame, end_frame, data_dict):
        "Find all tracks within the selected frames and return their track IDs"

        all_ids1 = data_dict[start_frame][:, -1]
        all_ids2 = data_dict[end_frame][:, -1]

        ids_sel = np.intersect1d(all_ids1, all_ids2)

        return ids_sel


    def find_all_track_candidates_from_end_frame(self, end_frame, data_dict):
        "Find all tracks within the selected end frame and return their track IDs"

        ids_sel = data_dict[end_frame][:, -1]
        real_det = np.sum(data_dict[end_frame][:, :self.cfg.MOT_INPUT_DIM], axis=1) != 0
        assert(ids_sel.shape[0] == real_det.shape[0])
        ids_sel = ids_sel[real_det]
        assert(ids_sel.shape[0] == np.sum(real_det))
        
        return ids_sel



class Real_tracking_episode(Data_generator):

    def __init__(self, seq_list, dataset, network, cfg):

        super(Real_tracking_episode, self).__init__(seq_list, dataset, network, cfg)

    def generate_tracking_episode(self):
        
        while True:
            # training parameters
            is_sequence_flipping = False # not used when generating real tracking episodes
            noise_sel = self.get_detection_noise() # select jittering noise

            seq_obj, seq_data, batch_info = self.get_tracks()
            mini_batch = self.initialize_batch(batch_info)
            image_cache = {}

            assert(batch_info['batch_num'] == np.shape(batch_info['ids_sel'])[0])
        
            # Track examples. 0 represents missing detection or no detection (when the object doesn't exist in the frame) 
            # ---000------ track ----000------
            # 000 --00---- track -----00------
            # ----0000---- track ----000-- 000
            # 000 -------- track ----000-- 000
            for i in range(0, batch_info['batch_num']):
                trk_id = batch_info['ids_sel'][i]
                trk_info = self.get_track_sync_info(trk_id, batch_info, seq_data)
                det_count = self.initialize_detection_count()
                self.initialize_norm_bboxes_prev(i, mini_batch, batch_info)

                is_missing_detection_aug = bool(np.random.rand(1) > (1 - self.cfg.MISSING_AUG_RATE)) 
                miss_det_rate = np.random.uniform(low=self.cfg.MISSING_DET_RATE_MIN, high=self.cfg.MISSING_DET_RATE_MAX, size=1)

                last_aug_missing_det_ct_tmp = 0
                last_missing_det_ct_tmp = 0
                iou_tmp = 0.0
                first_det_appeared = False
                # The track should exist in the selected time period 
                # Some tracks having no detection for the selected time period are fine 
                # if the track already started and didn't end yet 
                assert(batch_info['num_step'] - trk_info['start_offset1'] - trk_info['end_offset'] > 0)

                for j in range(0, batch_info['num_step'] - trk_info['start_offset1'] - trk_info['end_offset']):
                    # calculate the starting index where the first detection appears in the current track
                    ind_sel = j + trk_info['start_offset2']
                    # get the detection info from the selected index
                    bbox = self.get_bbox(seq_data['data_in_trk'], trk_id, ind_sel)
                    # initialize bbox_new
                    bbox_new = {'frame': 0}
                    # on/off flag for the missing detection augmentation 
                    miss_det_on = bool(np.random.rand(1) > (1 - miss_det_rate))
                    # check track sync info
                    self.error_check1(j, bbox['frame'], batch_info, trk_info)

                    assert(batch_info['end_frame_sel'] >= bbox['frame'])
                    # missing detection augmentation (i.e. randomly drop detections)
                    if batch_info['end_frame_sel'] != bbox['frame'] and \
                       trk_info['start_frame'] != bbox['frame'] and \
                       trk_info['end_frame'] != bbox['frame'] and \
                       is_missing_detection_aug == True and miss_det_on == True:
                        assert(trk_info['start_frame'] < bbox['frame'])
                        if first_det_appeared == False:       
                            # Even though this is missing detection due to augmentation,
                            # I treat it as the real missing detection to simplify the implementation
                            det_count['first_missing_det_ct'] += 1
                            det_count['missing_det_ct'] += 1
                            mini_batch['mot_frame_num'][i, j] = bbox['frame']
                            # bbox_new['frame'] = bbox['frame']
                        # Drop detections
                        else:       
                            # current frame missing detection due to missing detection augmentation
                            det_count['aug_missing_det_ct'] += 1
                            last_aug_missing_det_ct_tmp += 1
                            mini_batch['mot_frame_num'][i, j] = bbox['frame']
                            # bbox_new['frame'] = bbox['frame']
                    # no missing detection augmentation (i.e. read the detection info from the original track)
                    else:                                            
                        assert(seq_data['dtype'] in ['gt', 'det'])
                        # missing detection in the beginning of the track
                        if self.is_missing_detection(bbox) == True and first_det_appeared == False:
                            det_count['first_missing_det_ct'] += 1
                            det_count['missing_det_ct'] += 1
                            mini_batch['mot_frame_num'][i, j] = bbox['frame']   
                            # bbox_new['frame'] = bbox['frame']
                            continue
                        # missing detection in the middle of the track
                        elif self.is_missing_detection(bbox) == True and first_det_appeared == True:
                            last_missing_det_ct_tmp += 1
                            det_count['mid_missing_det_ct'] += 1
                            det_count['missing_det_ct'] += 1
                            mini_batch['mot_frame_num'][i, j] = bbox['frame']   
                            # bbox_new['frame'] = bbox['frame']
                            continue
                        # True positive detection
                        else:
                            first_det_appeared = True
                            last_missing_det_ct_tmp = 0
                            last_aug_missing_det_ct_tmp = 0
                            det_count['real_det_ct'] += 1

                        # add random noise in bbox coordinates
                        bbox_new, iou_tmp = self.add_bb_noise(bbox, batch_info['img_sz'], noise_sel, seq_data['dtype'])                                                            

                        miss_det_offset = int(det_count['aug_missing_det_ct'] + det_count['missing_det_ct'])
                        det_count['miss_det_offset'] = miss_det_offset
                        assert((j - det_count['miss_det_offset']) == (det_count['real_det_ct'] - 1))
                        assert(bbox['frame'] == bbox_new['frame'])

                        self.update_tracks_in_batch(
                            (i, j),
                            mini_batch,
                            batch_info,
                            seq_obj,
                            image_cache,
                            bbox,
                            bbox_new,
                            is_sequence_flipping,
                            det_count
                        )

                det_count['aug_missing_det_ct'] -= last_aug_missing_det_ct_tmp
                det_count['mid_missing_det_ct'] -= last_missing_det_ct_tmp
                det_count['missing_det_ct'] += last_aug_missing_det_ct_tmp
                det_count['last_missing_det_ct'] = (last_missing_det_ct_tmp + last_aug_missing_det_ct_tmp)

                assert(bbox['frame'] == trk_info['end_frame_in_minibatch'])
                if (first_det_appeared == False) and (det_count['first_missing_det_ct'] == det_count['missing_det_ct']):
                    det_count['last_missing_det_ct'] = det_count['first_missing_det_ct'] 
                    det_count['first_missing_det_ct'] = 0
                    assert(trk_info['start_offset1'] == 0 and trk_info['end_offset'] == 0 and trk_info['start_frame'] < batch_info['start_frame_sel'])

                # used bbox_new['frame'] instead of bbox['frame'] to get the frame # of the last real detection
                self.update_other_info_in_batch(i, mini_batch, batch_info, trk_info, det_count, iou_tmp, bbox_new['frame'])
                self.error_check2(i, mini_batch, batch_info, trk_info, det_count, is_missing_detection_aug)

            self.error_check3(mini_batch)    
            break
    
        return self.preprocess_track_data(mini_batch, batch_info)

    def preprocess_track_data(self, mini_batch, batch_info):

        # prepare input to the network in the right format
        images_sz = (batch_info['batch_num'] * batch_info['num_step'], self.cfg.IMAGE_HEIGHT, self.cfg.IMAGE_WIDTH, 3)
        mini_batch['images'] = mini_batch['images'].astype(np.float32)
        mini_batch['images'] = np.reshape(mini_batch['images'], images_sz)
        mini_batch['norm_bboxes'] = np.reshape(mini_batch['norm_bboxes'], (-1, self.cfg.MOT_INPUT_DIM))
        mini_batch['norm_bboxes_prev'] = np.reshape(mini_batch['norm_bboxes_prev'], (-1, self.cfg.MOT_INPUT_DIM))
        mini_batch['orig_bboxes_synced_with_app'] = np.reshape(mini_batch['orig_bboxes_synced_with_app'], (-1, self.cfg.MOT_INPUT_DIM))
        mini_batch['orig_bboxes_synced_with_mot'] = np.reshape(mini_batch['orig_bboxes_synced_with_mot'], (-1, self.cfg.MOT_INPUT_DIM))
        mini_batch['orig_noisy_bboxes_synced_with_app'] = np.reshape(mini_batch['orig_noisy_bboxes_synced_with_app'], (-1, self.cfg.MOT_INPUT_DIM))
        mini_batch['orig_noisy_bboxes_synced_with_mot'] = np.reshape(mini_batch['orig_noisy_bboxes_synced_with_mot'], (-1, self.cfg.MOT_INPUT_DIM))
        mini_batch['orig_noisy_bboxes_prev_synced_with_mot'] = np.reshape(mini_batch['orig_noisy_bboxes_prev_synced_with_mot'], (-1, self.cfg.MOT_INPUT_DIM))

        # data centering and normalization
        mini_batch['images'] -= np.array(self.cfg.IMAGENET_MEAN)
        mini_batch['norm_bboxes'] -= np.array(self.cfg.MOTION_MEAN)
        mini_batch['norm_bboxes'] /= np.array(self.cfg.MOTION_STD)
        mini_batch['norm_bboxes_prev'] -= np.array(self.cfg.MOTION_MEAN)
        mini_batch['norm_bboxes_prev'] /= np.array(self.cfg.MOTION_STD)

        # convert from BGR to RGB
        mini_batch['images'] = mini_batch['images'][:, :, :, [2, 1, 0]]

        # provide frames that will be used for generating matching candidates 
        # Currently, I only use the detections in the last frame
        frame_ind_sel = np.array([batch_info['num_step'] - 1])
        sample_frame_num = 1
        assert(sample_frame_num == self.cfg.SAMPLE_NUM_FOR_LOSS)

        # provide indices for building memory that contains negative target information
        neg_mem_indices = build_neg_mem_indices(batch_info['batch_num'], batch_info['batch_num'])

        # generate a mask for ignoring training examples in which an invalid track or an invalid detection is selected
        sel_indices = self.get_sel_indices(mini_batch, batch_info, sample_frame_num)

        np_indices = np.reshape(np.arange(batch_info['batch_num'] * batch_info['num_step'], dtype=np.int32), (-1, 1))
        app_shape = np.array([batch_info['batch_num'] * batch_info['num_step'], self.cfg.APP_LAYER_DIM], dtype=np.int32)
        mot_shape = np.array([batch_info['batch_num'] * batch_info['num_step'], self.cfg.MOT_INPUT_DIM], dtype=np.int32)

        mini_batch['end_offset_binary'] = mini_batch['end_offset'].astype(bool)
        mini_batch['end_offset_binary'] = mini_batch['end_offset_binary'].astype(int)

        training_data = {
            'images': mini_batch['images'], # images (tf variable name)
            'norm_bboxes': mini_batch['norm_bboxes'], # boundingboxes
            'norm_bboxes_prev': mini_batch['norm_bboxes_prev'], # boundingboxes_prev
            'valid_app_data': mini_batch['valid_app_data'], # valid_data
            'valid_mot_data': mini_batch['valid_mot_data'], # valid_motion_data
            'start_offset': mini_batch['start_offset'], # s_offset
            'end_offset': mini_batch['end_offset'], # e_offset
            'end_offset_binary': mini_batch['end_offset_binary'], # e_offset_binary
            'mid_missdet_num': mini_batch['mid_miss_det_ct'],
            'first_missdet_num': mini_batch['first_miss_det_ct'],
            'aug_miss_det_num': mini_batch['aug_miss_det_ct'], # miss_det_aug_num
            'miss_det_num': mini_batch['miss_det_ct'], # miss_det_aug_num
            'is_started': mini_batch['is_started'],
            'track_len': mini_batch['trk_len'],
            'iou': mini_batch['iou'],
            'app_frame_num': mini_batch['app_frame_num'],
            'app_prev_frame_num': mini_batch['app_prev_frame_num'],
            'mot_frame_num': mini_batch['mot_frame_num'],
            'trk_bbox_frame': mini_batch['trk_bbox_frame'],
            'det_bbox_frame': mini_batch['det_bbox_frame'],
            'orig_bboxes_synced_with_app': mini_batch['orig_bboxes_synced_with_app'],
            'orig_bboxes_synced_with_mot': mini_batch['orig_bboxes_synced_with_mot'],
            'orig_noisy_bboxes_synced_with_app': mini_batch['orig_noisy_bboxes_synced_with_app'],
            'orig_noisy_bboxes_synced_with_mot': mini_batch['orig_noisy_bboxes_synced_with_mot'],
            'orig_noisy_bboxes_prev_synced_with_mot': mini_batch['orig_noisy_bboxes_prev_synced_with_mot'],
            'lstm_state_frame': mini_batch['lstm_state_frame'],
            'last_orig_bbox': mini_batch['last_orig_bbox'],
            'all_ids': batch_info['ids_sel'],
            'num_step': batch_info['num_step'],
            'batch_num': batch_info['batch_num'],
            'is_last_frame': batch_info['is_last_frame'],
            'seq_name': batch_info['seq_name'],
            'detector': batch_info['detector'],
            'start_frame_sel': batch_info['start_frame_sel'], # s_frame
            'frame_ind_sel': frame_ind_sel, # sel_frames
            'neg_mem_indices':  neg_mem_indices, # neg_indices
            'sel_indices': sel_indices, # ignore_indicies
            'mapping_indices': np_indices,
            'image_batch_shape': app_shape,
            'bbox_batch_shape': mot_shape
        }

        return training_data

    def initialize_detection_count(self):
        
        det_count = {
            'aug_missing_det_ct': 0, # the total # of augmented missing detections in the current sequence
            'missing_det_ct': 0, # the total # of missing detections in the current sequence
            'first_missing_det_ct': 0, # the # of missing detections in the beginning of current sequence
            'last_missing_det_ct': 0, # the # of missing detections in the end of current sequence
            'mid_missing_det_ct': 0, # the # of missing detections in the middle of current sequence
            'real_det_ct': 0 # the # of actual detections in the current sequence
        }

        return det_count

    def update_tracks_in_batch(
            self, 
            inds,
            mini_batch,
            batch_info,
            seq_obj,
            image_cache,
            bbox,
            bbox_new,
            is_seq_flip,
            det_count
        ):

        i, j = inds
        trk_id = batch_info['ids_sel'][i]
        app_time_idx = j - det_count['miss_det_offset']
        mot_time_idx = j

        if self.network not in self.motion_network_list:
            roi_img = seq_obj.get_roi_in_image(image_cache, bbox_new, is_seq_flip)
            cnn_input_size = (self.cfg.IMAGE_WIDTH, self.cfg.IMAGE_HEIGHT)
            mini_batch['images'][i, app_time_idx, :, :, :] = cv2.resize(roi_img, cnn_input_size)
        else:
            assert(self.network in self.motion_network_list) 

        norm_bbox = self.get_motion_lstm_input(bbox_new, batch_info['img_sz'], is_seq_flip)
        noisy_orig_bbox = self.get_bbox_orig(bbox_new)
        orig_bbox = self.get_bbox_orig(bbox)
        mini_batch['norm_bboxes'][i, mot_time_idx, :] = norm_bbox
        mini_batch['norm_bboxes_prev'][i, mot_time_idx, :] = self.get_prev_valid_bbox(batch_info['prev_valid_norm_bbox_tmp'], trk_id, norm_bbox) 
        # store this for checking sync errors
        mini_batch['orig_bboxes_synced_with_app'][i, app_time_idx, :] = orig_bbox
        mini_batch['orig_bboxes_synced_with_mot'][i, mot_time_idx, :] = orig_bbox
        mini_batch['orig_noisy_bboxes_synced_with_app'][i, app_time_idx, :] = noisy_orig_bbox
        mini_batch['orig_noisy_bboxes_synced_with_mot'][i, mot_time_idx, :] = noisy_orig_bbox
        mini_batch['orig_noisy_bboxes_prev_synced_with_mot'][i, mot_time_idx, :] = self.get_prev_valid_bbox(batch_info['prev_valid_bbox_tmp'], trk_id, noisy_orig_bbox) 

        mini_batch['valid_app_data'][i, app_time_idx, 0] = 1.0
        mini_batch['valid_mot_data'][i, mot_time_idx, 0] = 1.0
        mini_batch['app_frame_num'][i, app_time_idx] = bbox['frame']
        mini_batch['app_prev_frame_num'][i, app_time_idx] = self.get_prev_valid_frame(batch_info['prev_valid_frame_tmp'], trk_id, bbox['frame'])
        mini_batch['mot_frame_num'][i, mot_time_idx] = bbox['frame']

        # trk_bbox_frame for debugging. 
        # currently, when the most recent frame of the track is located before this mini-batch, mini_batch['trk_bbox_frame'][i] is set to 0.
        # Thus, I am not using (can't use) trk_bbox_frame to verify the track frame number in that case.
        prev_frame = self.get_prev_valid_frame(batch_info['prev_valid_frame_tmp'], trk_id, bbox['frame'])
        cur_frame = bbox['frame']
        if bbox['frame'] == batch_info['end_frame_sel'] and det_count['real_det_ct'] > 1:
            mini_batch['trk_bbox_frame'][i] = prev_frame
        elif bbox['frame'] < batch_info['end_frame_sel']:
            mini_batch['trk_bbox_frame'][i] = cur_frame
        assert(mini_batch['trk_bbox_frame'][i] < batch_info['end_frame_sel'])
        self.update_prev_valid_tmp_dict(batch_info, trk_id, norm_bbox, noisy_orig_bbox, bbox['frame'])

        if bbox['frame'] < batch_info['end_frame_sel']:
            mini_batch['norm_bboxes_prev'][i, (mot_time_idx + 1):, :] = norm_bbox
            mini_batch['orig_noisy_bboxes_prev_synced_with_mot'][i, (mot_time_idx + 1):, :] = noisy_orig_bbox
            mini_batch['app_prev_frame_num'][i, (app_time_idx + 1):] = cur_frame
            assert((j + 1) <= (batch_info['num_step'] - 1))
        assert((j - det_count['miss_det_offset']) == (det_count['real_det_ct'] - 1))

        if (bbox['frame'] == batch_info['start_frame_sel']) and (batch_info['end_frame_sel'] - batch_info['start_frame_sel'] == (self.cfg.TB_SEQ_LEN - 1)):
            assert(trk_id not in batch_info['is_prev_valid_initialized'].keys())
            self.update_prev_valid_dict(batch_info, trk_id, norm_bbox, noisy_orig_bbox, bbox['frame'])
            batch_info['is_prev_valid_initialized'][trk_id] = True

    def update_other_info_in_batch(self, i, mini_batch, batch_info, trk_info, det_count, iou, frame):

        mini_batch['iou'][i] = iou
        mini_batch['trk_len'][i] = batch_info['num_step'] - trk_info['start_offset1'] - trk_info['end_offset'] - det_count['missing_det_ct'] - det_count['aug_missing_det_ct'] # the # of detections in the current sequence
        mini_batch['start_offset'][i] = trk_info['start_offset1'] + det_count['first_missing_det_ct'] # the # of frames that don't have detections in the begining of sequence 
        mini_batch['end_offset'][i] = trk_info['end_offset'] + det_count['last_missing_det_ct'] # the # of frames that don't have detections in the end of sequence
        mini_batch['miss_det_ct'][i] = det_count['missing_det_ct'] # the total # of missing detections 
        mini_batch['mid_miss_det_ct'][i] = det_count['mid_missing_det_ct'] # the # of missing detections in the middle of sequence
        mini_batch['first_miss_det_ct'][i] = det_count['first_missing_det_ct'] # the # of missing detections in the beginning of sequence
        mini_batch['aug_miss_det_ct'][i] = det_count['aug_missing_det_ct'] # the total # of missing detections augmented
        mini_batch['is_started'][i] = (trk_info['start_frame'] < batch_info['start_frame_sel'])

        if frame == batch_info['end_frame_sel']:
            assert(batch_info['end_frame_sel'] == trk_info['end_frame_in_minibatch'])
            mini_batch['det_bbox_frame'][i] = batch_info['end_frame_sel']
        else:
            assert(frame < batch_info['end_frame_sel'])
            # in case of missing detection in the end frame, end_frame_in_minibatch == end_frame_sel
            # otherwise, trach has ended before the end_frame. end_frame_in_minibatch < end_frame_sel
            assert(trk_info['end_frame_in_minibatch'] <= batch_info['end_frame_sel'])

    def initialize_norm_bboxes_prev(self, i, mini_batch, batch_info):

        track_id = batch_info['ids_sel'][i]
        prev_valid_bbox = batch_info['prev_valid_bbox']
        prev_valid_norm_bbox = batch_info['prev_valid_norm_bbox']
        prev_valid_frame = batch_info['prev_valid_frame']

        if track_id in prev_valid_norm_bbox.keys():
            assert(track_id in prev_valid_frame.keys())
            assert(track_id in prev_valid_bbox.keys())
            mini_batch['norm_bboxes_prev'][i, :, :] += prev_valid_norm_bbox[track_id]
            mini_batch['orig_noisy_bboxes_prev_synced_with_mot'][i, :, :] += prev_valid_bbox[track_id]
            mini_batch['app_prev_frame_num'][i, :] += prev_valid_frame[track_id]
            mini_batch['last_orig_bbox'][i, :] = prev_valid_bbox[track_id]
            mini_batch['lstm_state_frame'][i] = prev_valid_frame[track_id]
            batch_info['prev_valid_norm_bbox_tmp'][track_id] = prev_valid_norm_bbox[track_id]
            batch_info['prev_valid_bbox_tmp'][track_id] = prev_valid_bbox[track_id]
            batch_info['prev_valid_frame_tmp'][track_id] = prev_valid_frame[track_id]
            assert(prev_valid_frame[track_id] < batch_info['start_frame_sel'])
        else:
            assert(track_id not in prev_valid_frame.keys())
            assert(track_id not in prev_valid_bbox.keys())

    def get_prev_valid_bbox(self, prev_valid_bbox, track_id, bbox):

        if track_id not in prev_valid_bbox.keys():
            prev_valid_bbox[track_id] = bbox

        return prev_valid_bbox[track_id]

    def get_prev_valid_frame(self, prev_valid_frame, track_id, frame):

        if track_id not in prev_valid_frame.keys():
            prev_valid_frame[track_id] = frame

        return prev_valid_frame[track_id]

    def update_prev_valid_tmp_dict(self, batch_info, track_id, norm_bbox, orig_bbox, frame):

        prev_valid_norm_bbox = batch_info['prev_valid_norm_bbox_tmp']
        prev_valid_bbox = batch_info['prev_valid_bbox_tmp']
        prev_valid_frame = batch_info['prev_valid_frame_tmp']

        assert(track_id in prev_valid_norm_bbox.keys())
        assert(track_id in prev_valid_frame.keys())
        prev_valid_norm_bbox[track_id] = norm_bbox
        prev_valid_bbox[track_id] = orig_bbox
        prev_valid_frame[track_id] = frame

    def update_prev_valid_dict(self, batch_info, track_id, norm_bbox, orig_bbox, frame):

        prev_valid_norm_bbox = batch_info['prev_valid_norm_bbox']
        prev_valid_bbox = batch_info['prev_valid_bbox']
        prev_valid_frame = batch_info['prev_valid_frame']

        prev_valid_norm_bbox[track_id] = norm_bbox
        prev_valid_bbox[track_id] = orig_bbox
        prev_valid_frame[track_id] = frame

    def error_check1(self, j, frame, batch_info, trk_info):

        # check to see if the offsets are valid
        assert(frame == (batch_info['start_frame_sel'] + trk_info['start_offset1'] + j))
        assert(frame <= trk_info['end_frame_in_minibatch'] and frame <= trk_info['end_frame'])
        if j != ((batch_info['num_step'] - trk_info['start_offset1'] - trk_info['end_offset']) - 1):
            assert(frame != trk_info['end_frame'] and frame != trk_info['end_frame_in_minibatch'])
        else:
            assert(frame == trk_info['end_frame'] or frame == trk_info['end_frame_in_minibatch'])

    def error_check2(self, i, mini_batch, batch_info, trk_info, det_count, is_missing_detection_aug):

        # check to see if the sequence info is all correct
        # assert(training_trlen[i] != 0)
        # if seq_data['dtype'] == 'gt':
        #     assert(training_miss_det_ct[i] == 0)
        #     assert(last_missing_det_ct == 0)
        #     assert(first_missing_det_ct == 0)
        #     assert(mid_missing_det_ct == 0)

        # check to see if the sequence info is all correct
        end_ind = int(batch_info['num_step'] - trk_info['start_offset1'] - trk_info['end_offset'])
        assert(end_ind > 0)
        assert(np.sum(np.diff(mini_batch['mot_frame_num'][i, :end_ind])) == \
            (np.shape(mini_batch['mot_frame_num'][i, :end_ind])[0] - 1))
        assert(np.sum(mini_batch['mot_frame_num'][i, end_ind:]) == 0)

        assert(batch_info['num_step'] == trk_info['start_offset1'] + trk_info['end_offset'] + det_count['missing_det_ct'] + det_count['real_det_ct'] + det_count['aug_missing_det_ct'])
        assert(det_count['missing_det_ct'] == (det_count['first_missing_det_ct'] + det_count['mid_missing_det_ct'] + det_count['last_missing_det_ct']))
        assert(det_count['last_missing_det_ct'] >= 0 and det_count['first_missing_det_ct'] >= 0)
        assert(mini_batch['trk_len'][i] == det_count['real_det_ct'])
        if det_count['last_missing_det_ct'] > 0:
            assert(trk_info['end_offset'] == 0)
        elif trk_info['end_offset'] > 0:
            assert(det_count['last_missing_det_ct'] == 0)
        else:
            assert(trk_info['end_offset'] == 0 and det_count['last_missing_det_ct'] == 0)

        if det_count['first_missing_det_ct'] > 0:
            assert(trk_info['start_offset1'] == 0)
        if trk_info['start_offset1'] > 0:
            assert(det_count['first_missing_det_ct'] == 0)
        if is_missing_detection_aug == False:
            assert(mini_batch['aug_miss_det_ct'][i] == 0)
        if mini_batch['trk_len'][i] == 0:
            assert(mini_batch['is_started'][i] == True)
            assert(mini_batch['end_offset'][i] == batch_info['num_step'])
            assert(mini_batch['iou'][i] == 0)
        if mini_batch['end_offset'][i] == batch_info['num_step']:
            assert(mini_batch['trk_len'][i] == 0)
            assert(mini_batch['start_offset'][i] == 0)

    def error_check3(self, mini_batch):

        assert(np.sum(np.sum(mini_batch['norm_bboxes_prev'], axis=(1,2)) == 0) == 0)

    def select_seq(self):

        is_last_frame = False
        num_step = self.get_training_sequence_len() # the number of frames used for training      

        cur_seq = self.seq_list[self.cur_seq_idx]
        seq_name = cur_seq.seq_name
        start_frame = cur_seq.start_frame
        end_frame = cur_seq.end_frame
        prev_valid_bbox = cur_seq.last_valid_bbox
        prev_valid_norm_bbox = cur_seq.last_valid_norm_bbox
        prev_valid_frame = cur_seq.last_valid_frame
        if end_frame - start_frame == (num_step - 1):
            new_start_frame = start_frame + 1
        else:
            new_start_frame = start_frame
        assert((end_frame - start_frame) <= (num_step - 1))
        new_end_frame = end_frame + 1
        cur_seq.start_frame = new_start_frame
        cur_seq.end_frame = new_end_frame
        seq_obj = self.dataset[seq_name]
        seq_data_dict = self.get_data_from_seq(seq_obj, seq_name, cur_seq.dtype, cur_seq.detector)
        
        assert((end_frame - start_frame) <= (num_step - 1))

        # reset num_step in the beginning of the sequence
        if (end_frame - start_frame) < num_step - 1:
            num_step = end_frame - start_frame + 1

        # move on to the next sequence when all frames in the current sequence are processed
        if end_frame == seq_data_dict['last_frame']:
            is_last_frame = True
            self.seq_list[self.cur_seq_idx] = get_initial_seq_tuple(self.dataset,
                                                                    seq_name,
                                                                    cur_seq.dtype,
                                                                    cur_seq.detector)
            self.increment_seq_idx()

        # reset the sequence pointer
        if self.cur_seq_idx == self.seq_list_len:
            self.reset_seq_idx()

        cur_batch_info = {
            'seq_name': seq_name,
            'detector': cur_seq.detector,
            'start_frame_sel': start_frame,
            'end_frame_sel': end_frame,
            'is_last_frame': is_last_frame,
            'num_step': num_step,
            'img_sz': (seq_obj.img_height, seq_obj.img_width),
            'prev_valid_bbox': prev_valid_bbox,
            'prev_valid_norm_bbox': prev_valid_norm_bbox,
            'prev_valid_frame': prev_valid_frame,
            'prev_valid_norm_bbox_tmp': {},
            'prev_valid_bbox_tmp': {},
            'prev_valid_frame_tmp': {},
            'is_prev_valid_initialized': {}
        }

        return (seq_obj, seq_data_dict, cur_batch_info)

    def get_training_sequence_len(self):
        
        # return the sequence length used for truncated backpropagation
        return self.cfg.TB_SEQ_LEN

    def get_tracks(self):

        while True:

            seq_obj, seq_data, batch_info = self.select_seq()

            # collect detections & tracks in the selected time window
            dets_sel_all = {}
            tracks_sel_all = {}
            all_ids = []
            last_frame = 0
            ct = 0
            for i in range(0, batch_info['num_step']):
                frame_sel = batch_info['start_frame_sel'] + i
                last_frame = batch_info['start_frame_sel'] + i
                try:
                    dets_sel = seq_data['data_in_frame'][frame_sel]
                    ids_sel = dets_sel[:, -1]
                    dets_sel_all[frame_sel] = dets_sel
                    tracks_sel_all[frame_sel] = ids_sel
                    all_ids = np.concatenate((all_ids, ids_sel))
                except:
                    ct += 1
                    continue
            assert(last_frame != 0)
            # selected window does not have any track
            if ct == batch_info['num_step']:
                # print(seq_data['seq_name'], batch_info['start_frame_sel'])
                continue                
            else:
                break

        all_uniq_ids = np.unique(all_ids)
        all_uniq_ids = all_uniq_ids.astype(int)
        batch_num = np.shape(all_uniq_ids)[0]

        batch_info['det_sel'] = dets_sel_all
        batch_info['trk_sel'] = tracks_sel_all 
        batch_info['ids_sel'] = all_uniq_ids
        batch_info['batch_num'] = batch_num

        return (seq_obj, seq_data, batch_info)

    def get_data_from_seq(self, data_sel, seq_name, dtype, detector):

        data_sel_info = {}
        if dtype == 'gt':
            assert(detector == None)
            data_sel_info['seq_name'] = seq_name
            data_sel_info['first_frame'] = data_sel.gt_first_frame
            data_sel_info['last_frame'] = data_sel.gt_last_frame
            data_sel_info['data_in_frame'] = data_sel.data_in_frame
            data_sel_info['data_in_trk'] = data_sel.data_in_trk
            # data_sel_info['missingdets_in_trk'] = data_sel.data_missingdets_in_trk
            data_sel_info['dtype'] = dtype
        elif dtype == 'det':
            assert(detector in ['dpm', 'frcnn', 'sdp'])
            data_sel_info['seq_name'] = seq_name
            data_sel_info['first_frame'] = data_sel.det_first_frame[detector]
            data_sel_info['last_frame'] = data_sel.det_last_frame[detector]
            data_sel_info['data_in_frame'] = data_sel.dets_in_frame[detector]
            data_sel_info['data_in_trk'] = data_sel.dets_in_trk[detector]
            # data_sel_info['missingdets_in_trk'] = data_sel.dets_missingdets_in_trk[detector]
            data_sel_info['dtype'] = dtype
        else:
            raise NotImplementedError
        return data_sel_info

    def get_sel_indices(self, mini_batch, batch_info, sample_num):

        track_num = batch_info['batch_num']
        detbb_num = batch_info['batch_num']

        assert(sample_num == 1)
        sel_inds = np.ones((track_num, detbb_num, sample_num))
        for i in range(track_num):
            for j in range(detbb_num):
                for k in range(sample_num):
                    # check to see if there is a detection
                    if (mini_batch['end_offset'][j] != 0) or \
                       (mini_batch['end_offset'][i] == 0 and mini_batch['trk_len'][i] == 1 and mini_batch['is_started'][i] == False):
                        sel_inds[i, j, k] = 0.0
                        if mini_batch['end_offset'][i] == 0 and mini_batch['trk_len'][i] == 1 \
                           and mini_batch['miss_det_ct'][i] == 0:
                            assert(mini_batch['start_offset'][i] == (batch_info['num_step'] - 1))
        return sel_inds
    
    def get_track_sync_info(self, track_id, batch_info, seq_data):

        # track start frame
        tr_s_frame = seq_data['data_in_trk'][track_id][0, -1]
        tr_e_frame = seq_data['data_in_trk'][track_id][-1, -1]
        # calculate the offset to locate track data for the current frame 
        s_offset1 = tr_s_frame - batch_info['start_frame_sel'] # for tracks starting after the start frame
        s_offset2 = batch_info['start_frame_sel'] - tr_s_frame # for tracks starting before the start frame
        s_offset1 = int(np.maximum(s_offset1, 0))
        s_offset2 = int(np.maximum(s_offset2, 0))
        e_offset = (batch_info['start_frame_sel'] + batch_info['num_step'] - 1) - tr_e_frame
        e_offset = int(np.maximum(e_offset, 0))
        if s_offset2 > 0:
            assert(s_offset1 == 0)
        if s_offset1 > 0:
            assert(s_offset2 == 0)
        
        # the end frame in which the selected track ends in the current batch
        expected_end_frame = batch_info['start_frame_sel'] + (batch_info['num_step'] - e_offset - 1)
        # if the track's real end frame (i.e. when the object disappears from the scene) is in the current batch. 
        if e_offset > 0:
            assert(expected_end_frame == tr_e_frame)

        # information needed for synchronizing different tracks in a mini-batch
        track_sync_info = {
            'start_frame': tr_s_frame,
            'end_frame': tr_e_frame,
            'start_offset1': s_offset1,
            'start_offset2': s_offset2,
            'end_offset': e_offset,
            'end_frame_in_minibatch': expected_end_frame
        }

        return track_sync_info



# other functions 
def calculate_IoU(bb, bb_all):

    ymin, xmin, ymax, xmax = bb
    ymin_all, xmin_all, ymax_all, xmax_all = bb_all

    area = (xmax - xmin)*(ymax - ymin)
    area_all = np.multiply(xmax_all - xmin_all, ymax_all - ymin_all)

    x1 = np.maximum(xmin, xmin_all)
    y1 = np.maximum(ymin, ymin_all)
    x2 = np.minimum(xmax, xmax_all)
    y2 = np.minimum(ymax, ymax_all)

    w = np.maximum(0.0, x2 - x1)
    h = np.maximum(0.0, y2 - y1)

    intersection = np.multiply(w, h)
    union = np.add(area, area_all) - intersection
    IoU = np.divide(intersection, union)

    return IoU


def load_mot(data_path, seqs, visibility_thr, type):

    print('loading %s data...' % type)
    
    data_all = {}
    for seq_name in seqs:
        data_all[seq_name] = Data_MOT(seq_name, data_path, type, visibility_thr)        
        print(seq_name)

    return data_all


def jitter_bb_with_matched_noise(y, x, h, w, noise, cfg):
    
    # print('%s is selected' % noise['detector'])

    delta_sampled = np.random.multivariate_normal(noise['mean'], noise['cov']/cfg.COV_DIV, 1)
    x_delta = delta_sampled[0][0]
    y_delta = delta_sampled[0][1]
    w_delta = delta_sampled[0][2]
    h_delta = delta_sampled[0][3]

    x_new = x - x_delta*w
    y_new = y - y_delta*h
    w_new = w - w_delta*w
    h_new = h - h_delta*h

    return (y_new, x_new, h_new, w_new)


def jitter_bb(y, x, h, w, std):

    noise_mean = 0.0
    noise_std = std

    noise = np.random.normal(noise_mean, noise_std, 4)

    x = x + w * noise[0]
    y = y + h * noise[1]
    w = w * (1 + noise[2])
    h = h * (1 + noise[3])

    return (y, x, h, w)


def build_neg_mem_indices(track_num, detbb_num):
    
    neg_mem_ind = np.zeros((track_num, detbb_num, track_num-1, 2))

    if track_num > 1:
        for i in range(track_num):
            for j in range(detbb_num):
                xy_ind_tmp = np.zeros((track_num - 1, 2))
                x_ind_tmp = np.arange(track_num, dtype=np.int32)
                xy_ind_tmp[:, 0] = x_ind_tmp[x_ind_tmp != i]
                xy_ind_tmp[:, 1] = j
                neg_mem_ind[i, j, :, :] = xy_ind_tmp

    elif track_num == 1:
        neg_mem_ind = None
    else:
        raise NotImplementedError

    return neg_mem_ind

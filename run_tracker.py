import os
import cv2
import csv
import numpy as np
import config_tracker as cfg_trk
import config_train as cfg_tra
import tensorflow as tf

from data_util import Detections_MOT
from trackers import (Greedy_Tracker_APP_BLSTM,
                      Greedy_Tracker_MOT_LSTM,
                      Greedy_Tracker_APP_MOT)
from models import (app_gating_net_BLSTM,
                    app_gating_net_BLSTM_MTP,
                    mot_gating_net_LSTM,
                    app_mot_gating_net)


# Tracker Flags
tf.app.flags.DEFINE_string('model_path', None, 'File path to TF checkpoint files')
tf.app.flags.DEFINE_string('output_path', None, 'Output folder')
tf.app.flags.DEFINE_string('detector', None, 'detector_type')
tf.app.flags.DEFINE_string('network_type', None, 'Network type')
tf.app.flags.DEFINE_float('threshold', 0.5, 'classifier threshold')

# Training Flags 
# These are just needed for defining the model and do no affect the tracker's behavior.
tf.app.flags.DEFINE_float('weight_decay', 0.00004, 'The weight decay on the model weights.')
tf.app.flags.DEFINE_float('batch_norm_decay', 0.99, 'Batch Norm Decay.')

FLAGS = tf.app.flags.FLAGS


def draw_bb(im, dets, track_ids, frame, seq_name, colors, cfg):

    for i in range(0, dets.shape[0]):
        bbox = dets[i,:]

        x = int(bbox[0])
        y = int(bbox[1])
        w = int(bbox[2])
        h = int(bbox[3])

        cv2.rectangle(im, (x,y), (x+w,y+h), colors[track_ids[i]], 4)
        cv2.putText(im, str(track_ids[i]), (x,y -2), 0, 0.9, colors[track_ids[i]], 4) 

    img_output_dir = os.path.join(cfg.output_dir,  'imgs')
    if os.path.isdir(img_output_dir) == False:
        os.mkdir(img_output_dir)

    seq_folder = seq_name + "_" + cfg.detector
    if os.path.isdir(os.path.join(img_output_dir, seq_folder)) == False:
        os.mkdir(os.path.join(img_output_dir, seq_folder))
    cv2.imwrite(os.path.join(img_output_dir, seq_folder, '%05d.jpg' % frame), im)


def visualize_tracks(result, seq_name, cfg):

    print(seq_name + " Writing image files")

    if type(result) is np.ndarray:                
        result = np.reshape(result, (-1, 6))        
        # create a color map
        min_id = int(np.min(result[:, 1]))
        max_id = int(np.max(result[:, 1]))
        colors = {}
        for i in range(min_id, max_id+1):
            colors[i] = (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))

        if 'MOT17' in seq_name:
            img_dir = img_dir = os.path.join(cfg.DATASET_DIR, seq_name + '-' + cfg.detector, 'img1') 
        else:
            img_dir = img_dir = os.path.join(cfg.DATASET_DIR, seq_name, 'img1') 

        min_fr = int(np.min(result[:, 0]))
        max_fr = int(np.max(result[:, 0]))
        for i in range(min_fr, max_fr+1):
            img_path = os.path.join(img_dir, '%06d.jpg' % i)
            img = cv2.imread(img_path)            
            ind = result[:, 0] == i

            if np.sum(ind) == 0:
                continue

            det_sel = result[ind, 2:]
            track_ids = result[ind, 1].astype(np.int32)
            draw_bb(img, det_sel, track_ids, i, seq_name, colors, cfg)


def create_MOT_file(data, new_result, cfg):

    if os.path.isdir(cfg.output_dir) is False:
        os.mkdir(cfg.output_dir)

    print(data.seq_name + " Writing output file")

    output_file = os.path.join(cfg.output_dir, data.seq_name + '-' + cfg.detector + '.txt')
    with open(output_file, 'w') as track_file:
        writer = csv.writer(track_file, delimiter=',')

        start = 0
        for i in new_result.keys():

            track = new_result[i]
            for j in range(0, track.shape[0]):
                track_id = int(track[j,0])
                cur_frame = int(track[j,1])
                xmin = float(track[j,2])
                ymin = float(track[j,3])
                width = float(track[j,4])
                height = float(track[j,5])

                data = [cur_frame, track_id, xmin, ymin, width, height, -1, -1, -1, -1]
                writer.writerow(data)

                if start == 0:
                    result = np.array([cur_frame, track_id, xmin, ymin, width, height])
                    start = 1
                else:
                    tmp = (result, np.array([cur_frame, track_id, xmin, ymin, width, height]))
                    result = np.vstack(tmp)

    return result


def collect_tracks(result, data, bad_tracks):

    min_track_id = min(result, key=lambda x: x[3])
    min_track_id = min_track_id[3]
    max_track_id = max(result, key=lambda x: x[3])
    max_track_id = max_track_id[3]

    new_result = {}
    test = {}
    for i in range(min_track_id, max_track_id + 1):

        if i in bad_tracks:
            continue

        ind_sel = [k for k, v in enumerate(result) if v[3] == i]

        track_fr_det = {}
        min_fr = float('Inf')
        max_fr = -1
        for j in ind_sel:
            cur_frame, det_id, det_score, track_id = result[j]

            if det_id is not None:
                try:
                    print(test[(cur_frame, det_id)])
                    print(cur_frame, det_id, track_id)
                    raise
                except KeyError:
                    test[(cur_frame, det_id)] = track_id

            if det_id is None:
                track_fr_det[cur_frame] = None
            else:
                track_fr_det[cur_frame] = data.detections[cur_frame][det_id,:-1]

            if cur_frame < min_fr:
                min_fr = cur_frame
            if cur_frame > max_fr:
                max_fr = cur_frame

        xy_all = np.zeros((max_fr-min_fr+1,2))
        xy_all = np.ma.array(xy_all, mask=np.zeros(xy_all.shape))
        wh_all = np.zeros((max_fr-min_fr+1,2))
        wh_all = np.ma.array(wh_all, mask=np.zeros(wh_all.shape))
        frame_all = np.zeros((max_fr-min_fr+1,1))
        is_det = np.zeros(max_fr-min_fr+1, dtype=bool)

        frames = track_fr_det.keys()
        frame_cur = min_fr
        ct1 = 0
        ct2 = 0
        for j in range(0, xy_all.shape[0]):
            if track_fr_det[frame_cur] is not None:
                xy_all[j, :] = track_fr_det[frame_cur][:2]
                wh_all[j, :] = track_fr_det[frame_cur][2:4]
                is_det[j] = True
                ct1 = j
            else:
                xy_all[j, :] = np.nan
                wh_all[j, :] = np.nan
                ct2 = j

            frame_all[j,0] = frame_cur
            frame_cur += 1

        if ct1 < ct2:
            xy_all = xy_all[:(ct1+1), :]
            wh_all = wh_all[:(ct1+1), :]
            frame_all = frame_all[:(ct1+1), :]
            is_det = is_det[:(ct1+1)]
        
        xy_all = xy_all[is_det, :]
        wh_all = wh_all[is_det, :]
        frame_all = frame_all[is_det, :]

        track_id_vec = i*np.ones((xy_all.shape[0], 1))
        new_result[i] = np.concatenate((track_id_vec, frame_all, xy_all, wh_all), axis=1)

    return new_result


def find_bad_tracks(result, seq_num, cfg):
    """
    find bad tracks based on its detection score and length
    """
    min_track_id = min(result, key=lambda x: x[3])
    min_track_id = min_track_id[3]
    max_track_id = max(result, key=lambda x: x[3])
    max_track_id = max_track_id[3]

    bad_tracks = []
    for i in range(min_track_id, max_track_id + 1):
        ind_sel = [k for k, v in enumerate(result) if v[3] == i]
        score_avg = 0
        track_length = 0
        last_dummy_num = 0
        ct = 0
        for j in ind_sel:
            cur_frame, det_id, det_score, track_id = result[j]
            assert(i == track_id)
            if det_score is not None:
                score_avg += det_score
                last_dummy_num = 0
            else:
                ct -= 1
                last_dummy_num += 1
            track_length += 1
            ct += 1

        # print(i, min_track_id, max_track_id)
        track_length = track_length - last_dummy_num
        score_avg /= ct
        det_ratio = float(ct)/track_length

        if track_length <= cfg.track_length or score_avg < cfg.det_score_avg[seq_num] \
            or det_ratio < cfg.det_ratio:
            bad_tracks.append(i)

    return bad_tracks


def update_cfg_tracker(args, cfg):

    cfg.network_type = args.network_type
    cfg.detector = args.detector
    cfg.nn_gating_thr = args.threshold

    if cfg.SEQ_NAMES[0][:5] == 'MOT17':
        cfg.detector_specifier = '-' + cfg.detector
    else:
        cfg.detector_specifier = ''

    cfg.data_path = {}
    cfg.info_path = {}
    cfg.img_path = {}
    cfg.gt_path = {}
    for seq_name in cfg.SEQ_NAMES:

        if cfg.IS_TRACKTOR == True:
            cfg.data_path[seq_name] = os.path.join(cfg.DATASET_DIR, seq_name + cfg.detector_specifier, 'det', 'tracktor_prepr_det.txt')
        else:
            cfg.data_path[seq_name] = os.path.join(cfg.DATASET_DIR, seq_name + cfg.detector_specifier, 'det', 'det.txt')
        
        cfg.info_path[seq_name] = os.path.join(cfg.DATASET_DIR, seq_name + cfg.detector_specifier, 'seqinfo.ini')
        cfg.img_path[seq_name] = os.path.join(cfg.DATASET_DIR, seq_name + cfg.detector_specifier, 'img1')
        cfg.gt_path[seq_name] = os.path.join(cfg.DATASET_DIR, seq_name + cfg.detector_specifier, 'gt', 'gt.txt')

    # threshold for pruning detections based on its confidence score
    cfg.det_score = cfg.DET_SCORE_DICT[cfg.detector]

    cfg.is_nms = False

    if cfg.detector == 'DPM':        
        cfg.is_nms = True

    cfg.model_path = FLAGS.model_path
    cfg.output_dir = FLAGS.output_path

    return cfg


def update_cfg_train(cfg):

    cfg.MAX_STEP = 1

    return cfg


def get_detection_imgs(data, frame_num, cfg, img_sz):
    # load a full image and bounding box cooridnates
    img_path = os.path.join(data.img_path, '%06d.jpg' % frame_num)
    img = cv2.imread(img_path)
    det_bbs = data.detections[frame_num]
    det_normalized_bbs = data.norm_detections[frame_num]
    assert(np.shape(det_normalized_bbs)[1] == 5)
    det_normalized_bbs = det_normalized_bbs[:, :-1]
    assert(np.shape(det_normalized_bbs)[1] == 4)
    det_scores = det_normalized_bbs[:, -1]

    # container for all detection images
    img_height, img_width = img_sz
    det_imgs = np.zeros((det_bbs.shape[0], img_height, img_width, 3))

    for i in range(0, det_bbs.shape[0]):
        ymin = int(np.maximum(0.0, det_bbs[i, 1]))
        ymax = int(np.minimum(data.img_height, det_bbs[i, 1] + det_bbs[i, 3]))
        xmin = int(np.maximum(0.0, det_bbs[i, 0]))
        xmax = int(np.minimum(data.img_width, det_bbs[i, 0] + det_bbs[i, 2]))
        cropped_img = img[ymin:ymax, xmin:xmax, :]
        tmp_img = cv2.resize(cropped_img, (img_width, img_height))
        tmp_img = tmp_img - np.array([104, 117, 124])
        det_imgs[i, :, :, :] = tmp_img[:, :, [2, 1, 0]]    
    
    det_normalized_bbs -= np.array(cfg.MOTION_MEAN)
    det_normalized_bbs /= np.array(cfg.MOTION_STD)

    return (det_normalized_bbs, det_bbs, det_imgs, det_scores)


def run_greedy_tracker(data, tf_vars, cfg_tracker, cfg_train, session):

    tf_ops, tf_placeholders = tf_vars

    tracker_list = {
        'appearance_blstm_mtp': Greedy_Tracker_APP_BLSTM,
        'appearance_blstm': Greedy_Tracker_APP_BLSTM,
        'motion_lstm': Greedy_Tracker_MOT_LSTM,
        'appearance_motion_network': Greedy_Tracker_APP_MOT,
    }

    tracker = tracker_list[FLAGS.network_type]
    greedy_tracker = tracker(cfg_tracker, cfg_train, tf_ops, tf_placeholders, session)
    frames = data.detections.keys()
    for frame_num in range(1, data.seq_len+1):
        
        end_str = '\r'
        if frame_num == data.seq_len:
            end_str = '\n'
        progress = (100 * float(frame_num) / data.seq_len)
        print(data.seq_name + " Processing... %d%%" % progress, end=end_str)
        
        det_bbs = None
        det_norm_bbs = None
        det_imgs = None
        det_size = (cfg_tracker.IMAGE_HEIGHT, cfg_tracker.IMAGE_WIDTH)
        if frame_num in frames:
            det_norm_bbs, det_bbs, det_imgs, _ = get_detection_imgs(
                                                     data, 
                                                     frame_num, 
                                                     cfg_train,
                                                     img_sz=det_size)
        greedy_tracker.run(det_bbs, det_norm_bbs, det_imgs, frame_num)

        if frame_num == data.seq_len:
            result = greedy_tracker.get_result()
            break

    return result


def select_model(network_type, cfg_train, FLAGS):

    network_list = {
        'appearance_blstm_mtp': app_gating_net_BLSTM_MTP,
        'appearance_blstm': app_gating_net_BLSTM,
        'motion_lstm': mot_gating_net_LSTM,
        'appearance_motion_network': app_mot_gating_net
    }

    network = network_list[network_type]
    network_inst = network(FLAGS, cfg_train, 'test')
    preds = network_inst.predict()
    tf_vars, loaders = network_inst.collect_tracker_ops(preds[1])
    
    return (tf_vars, loaders['saver'])


def generate_output_files(
        data,
        seq_name,
        seq_num,
        result,        
        cfg_tracker
    ):

    # remove bad tracks
    bad_tracks = find_bad_tracks(result, seq_num, cfg_tracker)
    new_result = collect_tracks(result, data, bad_tracks)

    # save the final tracks in the MOT Challenge format
    vis_result = create_MOT_file(data, new_result, cfg_tracker)
    if cfg_tracker.IS_VIS is True:        
        visualize_tracks(vis_result, seq_name, cfg_tracker)


def main(_):

    cfg_tracker = update_cfg_tracker(FLAGS, cfg_trk)        
    cfg_train = update_cfg_train(cfg_tra)

    # the image sizes for training and testing should be the same
    assert(cfg_tracker.IMAGE_HEIGHT == cfg_train.IMAGE_HEIGHT)
    assert(cfg_tracker.IMAGE_WIDTH == cfg_train.IMAGE_WIDTH)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True

    with tf.Session(config=config) as sess:

        # construct a model
        tf_vars, loader = select_model(cfg_tracker.network_type, cfg_train, FLAGS)

        if os.path.isdir(cfg_tracker.output_dir) == False:
            os.mkdir(cfg_tracker.output_dir)                

        # initalize the model
        sess.run(tf.global_variables_initializer())
        loader.restore(sess, cfg_tracker.model_path)

        seq_num = 0
        seq_names = cfg_tracker.SEQ_NAMES
        for seq_name in seq_names:
            data = Detections_MOT(
                        seq_name=seq_name,
                        seq_num=seq_num,
                        cfg_tracker=cfg_tracker,
                        cfg_train=cfg_train
                    )
            
            result = run_greedy_tracker(data, tf_vars, cfg_tracker, cfg_train, sess)
            
            # save results
            generate_output_files(data, seq_name, seq_num, result, cfg_tracker)
            
            seq_num += 1       


if __name__ == '__main__':
    tf.app.run()

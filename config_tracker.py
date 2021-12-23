# test seqs
SEQ_NAMES = ['MOT17-01', 'MOT17-03', 'MOT17-06', 'MOT17-07',
             'MOT17-08', 'MOT17-12', 'MOT17-14']

# Please set your own path here
DATASET_DIR =  'YOUR_PATH/MOT17/test'

# # train seqs
# SEQ_NAMES = ['MOT17-02', 'MOT17-04', 'MOT17-05', 'MOT17-09',
#              'MOT17-10', 'MOT17-11', 'MOT17-13']

# # Please set your own path here
# DATASET_DIR =  'YOUR_PATH/MOT17/train'

# add motion gating based on simple motion cues
IS_NAIVE_GATING_ON = True
# use MOT public detection processed by Tracktor
IS_TRACKTOR = True
# generate image output
IS_VIS = False 

# input size
IMAGE_HEIGHT = 128
IMAGE_WIDTH = 64

# NMS threshold
overlap = 0.5 
# minimum track length
track_length = 5
# minimum average det score for each track
det_score_avg = [-100, -100, -100, -100, -100, -100, -100]
# minimum ratio of real detections to all detections in each track
det_ratio = 0.0 
# maximum number for consecutive missing detection for each track
N_miss_max = 90

# detection score threshold for pruning detections
DET_SCORE_DICT = {}
if IS_TRACKTOR == True:
    DET_SCORE_DICT['DPM'] = [-100, -100, -100, -100, -100, -100, -100]
    DET_SCORE_DICT['FRCNN'] = [-100, -100, -100, -100, -100, -100, -100]
    DET_SCORE_DICT['SDP'] = [-100, -100, -100, -100, -100, -100, -100]
else:
    # ablation study setting
    DET_SCORE_DICT['DPM'] = [-0.4, -0.4, -0.4, -0.4, -0.4, -0.4, -0.4]
    DET_SCORE_DICT['FRCNN'] = [0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4]
    DET_SCORE_DICT['SDP'] = [0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4]
DATASET = 'MOT'

MODEL_SAVE = 60168
# MODEL_SAVE = 17696

# training setting
IMAGE_HEIGHT = 128
IMAGE_WIDTH = 64
BATCH_NUM = 8
MAX_ITER = 370000
# MAX_ITER = 110000
VISIBILITY_THR = 0.25

# data augmentation
MISSING_DET_RATE_MIN = 0.05
MISSING_DET_RATE_MAX = 0.95
MISSING_AUG_RATE = 1.0
NEW_TRACK_AUG_RATE = 0.10

# real tracking episode setup
TB_SEQ_LEN = 10

# random tracking episode setup
MIN_STEP = 2
MAX_STEP = 40

# LSTM model parameters
MOT_LAYER_DIM = 64
APP_LAYER_DIM = 256
APP_HIDDEN_DIM = 2048
MOT_HIDDEN_DIM = 64
OUT_LAYER_DIM = 512
MOT_INPUT_DIM = 4
OUT_NUM = 2

# possibly remove the following variables from the config file and hard code them in the code 
# because there is not much need of changing them
DATA_FETCHER_NUM = 1
QUEUE_MAX_SZ = 15
COV_DIV = 2
JITTER_STD = 0.05 # std used when noise type is 'random' 
IOU_UPPER = 0.5
IOU_LOWER = 0.0
CONST = 10
SAMPLE_NUM_FOR_LOSS = 1

# set your own path to the dataset
DATA_PATH = 'YOUR_OWN_PATH'

# Data statistics
IMAGENET_MEAN = [104, 117, 124]

# Calculated from MOT Challege training data 
MOTION_MEAN = [3.13407819, 4.4261435, 1.90255835, 0.42575831]
MOTION_STD = [1.90437055, 2.41598032, 1.56631271, 0.4068819]

MOTION_MEAN = MOTION_MEAN[:MOT_INPUT_DIM]
MOTION_STD = MOTION_STD[:MOT_INPUT_DIM]

TRAIN_SEQS = ['MOT15-01', 'MOT15-02', 'MOT15-03', 
              'MOT15-04', 'MOT15-05', 'MOT15-06', 'MOT15-07', 'KITTI16', 
              'KITTI19',  'AVG-TownCentre', 'ETH-Jelmoli', 'ETH-Seq01',
              'PETS09-S2L2', 'TUD-Crossing']

VAL_SEQS = ['MOT17-02', 'MOT17-04', 'MOT17-05', 'MOT17-09', 'MOT17-10',
            'MOT17-11', 'MOT17-13']

# For bounding box jittering augmentation
DETECTOR_TYPE = ['DPM', 'FRCNN', 'SDP']


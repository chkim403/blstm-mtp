class Seq_data(object):

    def __init__(self, 
                 seq_name,
                 start_frame,
                 end_frame,
                 dtype,
                 detector):

        self.seq_name = seq_name
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.dtype = dtype
        self.detector = detector
        self.last_valid_frame = {}
        # this dictionary keeps the last valid bbox coordinates for each track
        self.last_valid_bbox = {}
        self.last_valid_norm_bbox = {}


def get_initial_seq_tuple(data, seq_name, dtype, detector):
    
    if dtype == 'gt':
        start_frame = data[seq_name].gt_first_frame
        end_frame = data[seq_name].gt_first_frame + 1
    elif dtype == 'det':
        start_frame = data[seq_name].det_first_frame[detector]
        end_frame = data[seq_name].det_first_frame[detector] + 1
    else:
        raise NotImplementedError

    return Seq_data(seq_name, 
                    start_frame,
                    end_frame,
                    dtype,
                    detector)


def generate_seq_iterator(data, det_type, include_det_tracks, is_train):

    seq_names = data.keys()
    iterator = []
    for seq_name in seq_names:
        seq_obj = get_initial_seq_tuple(data, seq_name, 'gt', None)
        iterator.append(seq_obj)

        if include_det_tracks == True and is_train == True and 'MOT17' in seq_name:
            seq_obj = get_initial_seq_tuple(data, seq_name, 'det', 'dpm')
            iterator.append(seq_obj)
            seq_obj = get_initial_seq_tuple(data, seq_name, 'det', 'frcnn')
            iterator.append(seq_obj)
            seq_obj = get_initial_seq_tuple(data, seq_name, 'det', 'sdp')
            iterator.append(seq_obj)
        elif include_det_tracks == True and is_train == True and 'MOT15' in seq_name:
            seq_obj = get_initial_seq_tuple(data, seq_name, 'det', 'dpm')
            iterator.append(seq_obj)

    return iterator
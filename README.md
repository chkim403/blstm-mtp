# BLSTM-MTP

This repository contains the official Tensorflow implementation of [Discriminative Appearance Modeling with Multi-track Pooling for Real-time Multi-object Tracking (CVPR 2021)](https://openaccess.thecvf.com/content/CVPR2021/html/Kim_Discriminative_Appearance_Modeling_With_Multi-Track_Pooling_for_Real-Time_Multi-Object_Tracking_CVPR_2021_paper.html).


## Dependencies
The code has been tested with:

- Python 3.7.6
- Tensorflow 1.15
- CUDA 10.0
- cuDNN 7.6.5
- OpenCV 4.5.3

## Download data

1. Download the MOT17 Challenge dataset from [this link](https://drive.google.com/file/d/1lZGLxWUcpRoVl0QGuPUx_ry10FqklaIf/view?usp=sharing). The zip file includes MOT Challenge public detections processed by [Tracktor](https://github.com/phil-bergmann/tracking_wo_bnw). We use this version of the public detections in our tracking demo below.
2. If you already downloaded the dataset from the official MOT Challenge [website](https://motchallenge.net/) before, please download the data from [this link](https://drive.google.com/file/d/1LSda5Z44qJZX9K50PrvXHf8exrPKHvRy/view?usp=sharing) instead which doesn't include the image files. 

## Demo

1. Set `DATASET_DIR` in `config_tracker.py` to your own directory where the dataset you download above is located. 
2. If you want to write the tracking output as images as well, set `IS_VIS` in `config_tracker.py` to `True`. Otherwise, leave it as it is.
3. Download the model file from [here](https://drive.google.com/file/d/1dOYof9N8RhFACS5dkr-gLQcb8eTOXupJ/view?usp=sharing) and unzip the file. Use the location where the checkpoint file is located as `model_path` in the command below. 
4. Run the following command. Use your own paths for `model_path` and `output_path`. As for `detector`, you can use one of `DPM`, `FRCNN`, and `SDP`. 
```
python run_tracker.py --model_path=YOUR_MODEL_FOLDER/model.ckpt --output_path=YOUR_OUTPUT_FOLDER  --detector=FRCNN --threshold=0.5 --network_type=appearance_motion_network
```
5. This command will generate the tracking result that is shown in Table 6 of our paper. You can use these [files](https://drive.google.com/file/d/1CKdjUIlHbfO304IYwsrdblQ_PXEtmYCm/view?usp=sharing) to verify your output files.

## Performance

When paired with [Tracktor](https://github.com/phil-bergmann/tracking_wo_bnw) or [CenterTrack](https://github.com/xingyizhou/CenterTrack), our method greatly improves the tracking performance in terms of IDF1 and IDS. 

| Method                | IDF1        | MOTA        | IDS         | MT          | ML          | Frag        | FP          | FN          | 
| --------------------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- |
| Tracktor++v2          | 55.1        | 56.3        | 1,987       | 21.1        | 35.3        | 3,763       | 8,866       | 235,449     |
| Ours + Tracktor++v2   | 60.5        | 55.9        | 1,188       | 20.5        | 36.7        | 4,185       | 8,663       | 238,863     |

The data file that you download in the instructions above also includes MOT Challenge detections processed by CenterTrack ('centertrack_prepr_det.txt'). In order to use it as input to the tracker, you can simply change 'run_tracker.py' in a way that it reads detections from 'centertrack_prepr_det.txt' instead of 'tracktor_prepr_det.txt'. The following is the result obtained by using the public detections processed by CenterTrack.

| Method                | IDF1        | MOTA        | IDS         | MT          | ML          | Frag        | FP          | FN          | 
| --------------------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- |
| CTTrackPub            | 59.6        | 61.5        | 2,583       | 26.4        | 31.9        | 4,965       | 14,076      | 200,672     |
| Ours + CTTrackPub     | 62.9        | 62.0        | 1,750       | 27.9        | 31.0        | 7,433       | 17,621      | 194,946     |

With NVIDIA TITAN Xp, the inference code runs at around 24 fps on the MOT17 Challenge test set (excluding time spent on I/O operations). 

## Training
The training code will be released soon in the future release. Stay tuned for more updates.

## License
The code is released under the MIT License.

## Contact
If you have any questions, please contact me at chkim@gatech.edu.

## Citation
```
@InProceedings{Kim_2021_CVPR,
    author    = {Kim, Chanho and Fuxin, Li and Alotaibi, Mazen and Rehg, James M.},
    title     = {Discriminative Appearance Modeling With Multi-Track Pooling for Real-Time Multi-Object Tracking},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2021},
    pages     = {9553-9562}
}
```
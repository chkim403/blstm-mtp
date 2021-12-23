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
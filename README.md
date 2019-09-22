# IPN (Image Projection Network): 3D to 2D Image Segmentation in OCTA images

This is an implementation of "Image Projection Network: 3D to 2D Image Segmentation in OCTA Images". IPN is proposed for 3D to 2D segmentation. Our key insight is to build a projection learning module(PLM) which uses a unidirectional pooling layer to conduct effective features selection and dimension reduction concurrently.By combining multiple PLMs, the proposed network can input 3D data and output 2D segmentation results.

# Paper

Image Projection Network: 3D to 2D Image Segmentation in OCTA Images

Mingchao Li, Yerui Chen, Shuo Li, Zexuan Ji, Keren Xie, Songtao Yuan, Qiang Chen

# Requirements

Ubuntu 18.04, Python 3.x, Tensorflow 1.12

# Test the code

python test.py

If success, the result will be find in 'logs/test_result/'.

# Train for your own data

## Data storage structure

Place the data as the following structure.(An example in 'dataset/test').

- dataset
    - train
        - image1(modality 1)
            - name1
                - 1.bmp
                - 2.bmp
                - 3.bmp
                - ...
            - name2
            - ...
        - image2(modality 1)
            - name1
                - 1.bmp
                - 2.bmp
                - 3.bmp
                - ...
            - name2
            - ...
        - ...
        - label
            - name1.bmp
            - name2.bmp
            - ...
    - test(same as train)
    - val(same as train)

## Train

python train.py

## Change Parameters

Parameters are set in 'options/'

The parameter description is in 'param_help.py'

## Model Saver

The best model is saved in 'logs/best_model/'

Other models are saved in 'logs/checkpoints/'

## Result

python test.py.

The results will be saved in 'logs/test_result'.

Some examples of our results and corresponding OCTA projection maps are in 'logs/other results/'.

![blockchain](https://github.com/chaosallen/IPN_master/blob/master/logs/test_results/10201.bmp)
![blockchain](https://github.com/chaosallen/IPN_master/blob/master/logs/test_results/10201-M-OS-44.bmp)

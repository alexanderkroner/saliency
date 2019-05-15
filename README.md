# Contextual Encoder-Decoder Network <br/> for Visual Saliency Prediction

![](https://img.shields.io/badge/python-v3.6.8-orange.svg?style=flat-square)
![](https://img.shields.io/badge/tensorflow-v1.13.1-orange.svg?style=flat-square)
![](https://img.shields.io/badge/matplotlib-v3.0.3-orange.svg?style=flat-square)
![](https://img.shields.io/badge/requests-v2.21.0-orange.svg?style=flat-square)

<img src="./figures/results.jpg" width="800"/>

This repository contains the official TensorFlow implementation of the MSI-Net (multi-scale information network), as described in the arXiv paper [Contextual Encoder-Decoder Network for Visual Saliency Prediction](https://arxiv.org/abs/1902.06634) (2019).

**_Abstract:_** *Predicting salient regions in natural images requires the detection of objects that are present in a scene. To develop robust representations for this challenging task, high-level visual features at multiple spatial scales must be extracted and augmented with contextual information. However, existing models aimed at explaining human fixation maps do not incorporate such a mechanism explicitly. Here we propose an approach based on a convolutional neural network pre-trained on a large-scale image classification task. The architecture forms an encoder-decoder structure and includes a module with multiple convolutional layers at different dilation rates to capture multi-scale features in parallel. Moreover, we combine the resulting representations with global scene information for accurately predicting visual saliency. Our model achieves competitive results on two public saliency benchmarks and we demonstrate the effectiveness of the suggested approach on selected examples. The network is based on a lightweight image classification backbone and hence presents a suitable choice for applications with limited computational resources to estimate human fixations across complex natural scenes.*

Our results on the MIT saliency benchmark can be viewed [here](http://saliency.mit.edu).

## Architecture

<img src="./figures/architecture.jpg" width="700"/>

## Requirements

| Package    | Version |
|:----------:|:-------:|
| python     | 3.6.8   |
| tensorflow | 1.13.1  |
| matplotlib | 3.0.3   |
| requests   | 2.21.0  |

The code was tested and is compatible with both Windows and Linux. We strongly recommend to use TensorFlow with gpu acceleration, especially when training the model. Nevertheless, a cpu version is officially supported.

## Training

The results of our paper can be reproduced by first training the MSI-Net via the following command:

```
python main.py train
```

This will start the training procedure for the SALICON dataset with the hyperparameters defined in `config.py`. If you want to optimize the model for cpu usage, please change the corresponding `device` value in the configurations file. Optionally, the dataset and download path can be specified via command line arguments:

```
python main.py train -d DATA -p PATH
```

Here, the `DATA` argument must be `salicon`, `mit1003`, or `cat2000`. It is required that the model is first trained on the SALICON dataset before fine-tuning it on either MIT1003 or CAT2000. By default, the selected saliency dataset will be downloaded to the folder `data/` but you can point to a different directory via the `PATH` argument.

All results are then stored under the folder `results/`, which contains the training history and model checkpoints. This allows to continue training or perform inference on test instances, as described in the next section.

## Testing

To test a pre-trained model on image data and produce saliency maps, execute the following command:

```
python main.py test -d DATA -p PATH
```

If no checkpoint is available from prior training, it will automatically download our pre-trained model to `weights/`. The `DATA` argument defines which network is used and must be `salicon`, `mit1003`, or `cat2000`. It will then resize the input images to the dimensions specified in the configurations file. Note that this might lead to excessive image padding depending on the selected dataset.

The `PATH` argument points to the folder where the test data is stored but can also denote a single image directly. Similar to network training, the `device` value can be changed to cpu in the configurations file. This ensures that the model optimized for cpu will be utilized and hence improves the inference speed. All results are then stored in the folder `results/images/` with the original image dimensions.

## Contact

For questions, bug reports, and suggestions about this work, please create an [issue](https://github.com/alexanderkroner/saliency/issues) in this repository.

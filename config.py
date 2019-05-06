"""General training parameters that define the maximum number of
   training epochs, the batch size, and learning rate for the ADAM
   optimization method. To reproduce the results from the paper,
   these values should not be changed. The device can be either
   "cpu" or "gpu", which then optimizes the model accordingly after
   training or uses the correct version for inference when testing.
"""

PARAMS = {
    "n_epochs": 10,
    "batch_size": 1,
    "learning_rate": 1e-5,
    "device": "gpu"
}

"""The predefined input image sizes for each of the 3 datasets.
   To reproduce the results from the paper, these values should
   not be changed. They must be divisible by 8 due to the model's
   downsampling operations. Furthermore, all pretrained models
   for download were trained on these image dimensions.
"""

DIMS = {
    "image_size_salicon": (240, 320),
    "image_size_mit1003": (360, 360),
    "image_size_cat2000": (216, 384)
}

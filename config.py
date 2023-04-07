from toolkit import *


def get_config():
    net = 'unet'
    ENCODER = 'vgg11bn'
    ENCODER_WEIGHTS = 'imagenet'
    CLASSES = ['0', '1', '2', '3', '4', '5', '6']  # 7 classes: background + 6 classes
    ACTIVATION = 'softmax2d'
    # create segmentation model with pretrained encoder
    model = smp.Unet(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        classes=len(CLASSES),
        activation=ACTIVATION,
    )
    return net, ENCODER, ENCODER_WEIGHTS, CLASSES, ACTIVATION, model

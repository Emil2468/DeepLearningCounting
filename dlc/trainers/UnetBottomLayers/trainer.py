#   Author: Emil Møller Hansen

from dlc.models.UnetBasedTrainer import UnetBasedTrainer
from .config import Config
from tensorflow.keras import layers, models


class Trainer(UnetBasedTrainer):

    def __init__(self, conf: Config):
        super().__init__(conf)

    def get_count_model(self, base_model):
        input_layer_idx = 20
        input_layer = base_model.layers[input_layer_idx]

        input_shape = input_layer.output.shape

        conv_layer = layers.Conv2D(int(input_shape[-1] / 64), (1, 1),
                                   name=f"count_conv_0",
                                   activation="elu")(input_layer.output)

        flat_layer = layers.Flatten()(conv_layer)
        dense_layer = layers.Dense(flat_layer.shape[1] / 512,
                                   activation="elu")(flat_layer)

        output_layer = layers.Dense(1)(dense_layer)

        counting_model = models.Model(inputs=base_model.inputs, outputs=[output_layer])
        self.compile_model(counting_model)
        return counting_model

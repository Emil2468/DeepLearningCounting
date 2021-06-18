#   Author: Emil Møller Hansen

from dlc.models.UnetBasedTrainer import UnetBasedTrainer
from .config import Config
from tensorflow.keras import layers, models


class Trainer(UnetBasedTrainer):

    def __init__(self, conf: Config):
        super().__init__(conf)

    def get_count_model(self, base_model):

        input_layer_idxs = [20, 25, 30, 35, 40]
        input_layers = [
            base_model.layers[input_layer_idx] for input_layer_idx in input_layer_idxs
        ]

        input_shapes = [input_layer.output.shape for input_layer in input_layers]

        conv_layers = [
            layers.Conv2D(int(input_shapes[idx][-1] / 64), (1, 1),
                          name=f"count_conv_{idx}",
                          activation="elu")(input_layer.output)
            for (idx, input_layer) in enumerate(input_layers)
        ]
        flat_layers = [layers.Flatten()(conv_layer) for conv_layer in conv_layers]
        dense_layers_1 = [
            layers.Dense(flat_layer.shape[1] / 512, activation="elu")(flat_layer)
            for flat_layer in flat_layers
        ]
        concat_layer = layers.concatenate(dense_layers_1)
        output_layer = layers.Dense(1)(concat_layer)
        counting_model = models.Model(inputs=base_model.inputs, outputs=[output_layer])
        self.compile_model(counting_model)
        return counting_model

#   Author: Emil Møller Hansen

from dlc.tools.datasets import ImageDatasetGenerator
import numpy as np
import pandas as pd
import tensorflow_addons as tfa
import numpy as np
from dlc.core.augmentation import DensityAugTransform0
from dlc.core.losses import (accuracy, dice_coef, dice_loss, sensitivity, specificity,
                             tversky)
from dlc.models.unet_nature_version import UNet
from dlc.tools.datasets import ImageDatasetGenerator
from dlc.tools.images import ImageCache, ImageLoader
from dlc.tools.splits import LatitudeObjectSplitter
from dlc.trainers import BaseTrainer
from tensorflow.keras.models import load_model
import tensorflow as tf
import geopandas as gpd


class UnetBasedTrainer(BaseTrainer):

    def __init__(self, conf):
        self.conf = conf
        self.patch_size = (self.conf.input_shape[1], self.conf.input_shape[2])
        if conf.whole_model_file_path is None:
            base_model = self.get_base_model()
            self.model = self.get_count_model(base_model)
        else:
            print('Loading model from file')
            self.model = load_model(self.conf.whole_model_file_path,
                                    custom_objects={
                                        'tversky': tversky,
                                        'dice_coef': dice_coef,
                                        'dice_loss': dice_loss,
                                        'accuracy': accuracy,
                                        'specificity': specificity,
                                        'sensitivity': sensitivity
                                    },
                                    compile=False)
            self.compile_model(self.model)
        self.model.summary()
        self.image_loader = ImageLoader(
            local_standardization_p=[0.4, None],
            seed=self.conf.seed,
            cache=ImageCache(),
        )
        self.eval_image_loader = ImageLoader(
            local_standardization_p=[0, None],
            seed=self.conf.seed,
            cache=ImageCache(),
        )

    def get_base_model(self):
        if self.conf.base_model_file_path is None:
            base_model = UNet(self.conf.input_shape,
                              self.conf.n_output_channels,
                              layer_count=self.conf.layer_count)
        else:
            base_model = load_model(self.conf.base_model_file_path,
                                    custom_objects={
                                        'tversky': tversky,
                                        'dice_coef': dice_coef,
                                        'dice_loss': dice_loss,
                                        'accuracy': accuracy,
                                        'specificity': specificity,
                                        'sensitivity': sensitivity
                                    },
                                    compile=False)
        base_model.trainable = not self.conf.freeze_base_model
        return base_model

    def get_count_model(self, base_model):
        pass

    def compile_model(self, model):
        model.compile(optimizer="adam",
                      loss="MSE",
                      metrics=[accuracy, tfa.metrics.RSquare(y_shape=(1,))])

    def _train(self):
        batch_size = self.conf.input_shape[0]
        image_augmenter = DensityAugTransform0()

        training_ds = self.images_ds_gen.get_sequential_patches(
            "training",
            self.patch_size,
            patch_stride=self.conf.patch_stride,
            seed=self.conf.seed,
            shuffle=True)
        training_ds = training_ds.map(self.image_loader.load,
                                      num_parallel_calls=tf.data.AUTOTUNE)
        training_ds = training_ds.map(to_count_annotation,
                                      num_parallel_calls=tf.data.AUTOTUNE)
        training_ds = training_ds.cache()
        training_ds = training_ds.map(image_augmenter,
                                      num_parallel_calls=tf.data.AUTOTUNE)
        training_ds = training_ds.batch(batch_size)
        training_ds = training_ds.prefetch(tf.data.AUTOTUNE)

        validation_ds = self.images_ds_gen.get_sequential_patches(
            "validation",
            self.patch_size,
            patch_stride=self.conf.patch_stride,
            shuffle=False,
            seed=self.conf.seed)
        validation_ds = validation_ds.map(self.eval_image_loader.load,
                                          num_parallel_calls=tf.data.AUTOTUNE)
        validation_ds = validation_ds.map(to_count_annotation,
                                          num_parallel_calls=tf.data.AUTOTUNE)
        validation_ds = validation_ds.cache()
        validation_ds = validation_ds.batch(batch_size)
        validation_ds = validation_ds.prefetch(tf.data.AUTOTUNE)

        self.checkpoint_path = f"{self.conf.model_save_path}{self.conf.model_name}.h5"

        print(f"Will save model checkpoints to {self.checkpoint_path}")
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            self.checkpoint_path,
            monitor="val_loss",
            verbose=1,
            save_best_only=True,
            mode="min",
            save_weights_only=False,
        )

        history = self.model.fit(
            training_ds,
            epochs=self.conf.n_epochs,
            validation_data=validation_ds,
            callbacks=[
                checkpoint_callback,
            ],
            verbose=1,
        )

    def _evaluate(self):
        batch_size = self.conf.input_shape[0]
        evaluation_ds = self.images_ds_gen.get_sequential_patches(
            "evaluation",
            self.patch_size,
            patch_stride=self.conf.patch_stride,
            shuffle=False,
            seed=self.conf.seed)
        evaluation_ds = evaluation_ds.map(self.eval_image_loader.load,
                                          num_parallel_calls=tf.data.AUTOTUNE)
        evaluation_ds = evaluation_ds.map(to_count_annotation,
                                          num_parallel_calls=tf.data.AUTOTUNE)
        evaluation_ds = evaluation_ds.cache()
        evaluation_ds = evaluation_ds.batch(batch_size)
        evaluation_ds = evaluation_ds.prefetch(tf.data.AUTOTUNE)

        result = self.model.evaluate(evaluation_ds)
        print(result)

    def evaluate(self):
        frames = gpd.read_file(self.conf.test_frames_file_path)
        splits = np.zeros(len(frames))
        splits_map = {"evaluation": 0}

        self.images_ds_gen = ImageDatasetGenerator(
            frames,
            splits=splits,
            splits_map=splits_map,
            input_base_path=self.conf.input_base_path,
            image_keys=(self.conf.input_image_keys, self.conf.output_image_keys),
        )
        self._evaluate()

    def run(self):
        frames = gpd.read_file(self.conf.frames_file_path)
        splitter = LatitudeObjectSplitter()
        splits, _ = splitter.run(frames, [self.conf.val_split], seed=self.conf.seed)
        splits_map = {"training": 0, "validation": 1}

        self.images_ds_gen = ImageDatasetGenerator(
            frames,
            splits=splits,
            splits_map=splits_map,
            input_base_path=self.conf.input_base_path,
            image_keys=(self.conf.input_image_keys, self.conf.output_image_keys),
        )

        self._train()
        if self.conf.test_frames_file_path:
            print(
                "Training done! Loading best model from previous epochs and evaluating..."
            )
            self.model = load_model(self.checkpoint_path,
                                    custom_objects={
                                        'tversky': tversky,
                                        'dice_coef': dice_coef,
                                        'dice_loss': dice_loss,
                                        'accuracy': accuracy,
                                        'specificity': specificity,
                                        'sensitivity': sensitivity
                                    },
                                    compile=False)
            self.compile_model(self.model)
            self._evaluate()

    def predict(self):
        batch_size = self.conf.input_shape[0]
        print(self.conf.frames_file_path)
        frames = gpd.read_file(self.conf.frames_file_path)

        images_ds_gen = ImageDatasetGenerator(
            frames,
            splits=np.zeros(len(frames)),
            splits_map={"pred": 0},
            input_base_path=self.conf.input_base_path,
            image_keys=self.conf.input_image_keys,
        )

        pred_image_loader = ImageLoader(
            local_standardization_p=[0],
            seed=self.conf.seed,
            cache=ImageCache(),
        )

        prediction_ds = images_ds_gen.get_sequential_patches("pred",
                                                             self.patch_size,
                                                             patch_stride=None,
                                                             shuffle=False,
                                                             seed=self.conf.seed)

        def load_image_and_return_path(image_loader):

            def load(spec):
                return image_loader.load(spec), spec["paths"]

            return load

        prediction_ds = prediction_ds.map(load_image_and_return_path(pred_image_loader),
                                          num_parallel_calls=tf.data.AUTOTUNE)
        prediction_ds = prediction_ds.cache()
        prediction_ds = prediction_ds.batch(batch_size)
        prediction_ds = prediction_ds.prefetch(tf.data.AUTOTUNE)
        dfs = []
        for inp, paths in tqdm(prediction_ds):
            preds = self.model.predict(inp)
            frame_names = [
                p.numpy().decode("utf-8").split("\\")[-1]
                for p in tf.reshape(paths, [-1])
            ]
            df = pd.DataFrame(data={
                "Frame": frame_names,
                "Predicted count": tf.reshape(preds, [-1])
            })
            dfs.append(df)

        pred_df = pd.concat(dfs)
        pred_df.to_csv(self.conf.output_file_path, index=False)


@tf.function
def to_count_annotation(x, y):
    return x, tf.expand_dims(tf.reduce_sum(y[:, :, 0]), 0)

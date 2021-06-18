#   Author: Emil Møller Hansen

from typing import Optional, Tuple, List


class Config():
    """Args:
            base_model_file_path
                File path to specification of base UNet model
            whole_model_file_path
                File path to the whole model
            input_shape: tuple(int, int, int, int)
                Shape of the input in the format (batch, patch_height, patch_width, channels).
            n_output_channels: int
                The numbe of output channels in the U-Net model, only relevant if base_model_file_path is None
            input_base_path: Optional[str]
                If frames contain relative paths, use this to define what path they are relative to.
            layer_count: (int, optional)
                Count of kernels in first layer. Number of kernels in other layers grows with a fixed factor.
            frames_file_path
                File path to training frames
            test_frames_file_path
                File path to testing/evaluation frames
            seed
                Seed used when splitting the training data set
            val_split
                Fraction of training data to use as validation set
            input_image_keys
                Keys of frame creators from pre-processing steps to use in input images
            output_image_keys
                Keys of frame creators from pre-processing steps to use in output images
            patch_stride
                Stride between each patch
            n_epochs
                Number of epochs to train the model for
            model_save_path
                Absolute path to folder to save the model in
            freeze_base_model
                Whether to freeze the U-net model or not
            output_file_path
                File to write predictions to
        """
    base_model_file_path: Optional[str] = None
    whole_model_file_path: Optional[str] = None
    input_shape: Tuple[int, int, int, int] = (8, 256, 256, 2)
    n_output_channels: int = 1
    input_base_path: Optional[str] = None
    layer_count: int = 64
    frames_file_path: str = ""
    test_frames_file_path: str = ""
    seed: int = 591477907
    val_split: float = 0.2
    input_image_keys: List[str] = ["image"]
    output_image_keys: List[str] = ["uniform-density"]
    patch_stride: Tuple[int, int] = (128, 128)
    n_epochs: int = 50
    model_save_path: str = ""
    freeze_base_model: bool = True
    output_file_path: str = ""

    def __init__(self) -> None:
        [setattr(self, k, v) for k, v in vars(Config).items() if not k.startswith("_")]

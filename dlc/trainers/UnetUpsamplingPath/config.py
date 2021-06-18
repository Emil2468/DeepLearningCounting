#   Author: Emil Møller Hansen

from dlc.models import UnetBasedTrainer


class Config(UnetBasedTrainer.Config):
    """Args:
            model_name
                Name of the file to save the model in. The model will be saved in `model_save_path/model_name.h5`
        """
    model_name: str = "unet_upsampling_path"

    def __init__(self) -> None:
        super().__init__()
        [setattr(self, k, v) for k, v in vars(Config).items() if not k.startswith("_")]

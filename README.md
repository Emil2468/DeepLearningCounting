# DeepLearningCounting
This repository contains code for the DLC-package (deep learning counting).
## Contributors
The authors of each file are denoted in the files.
## What is DLC?
DLC is a package intended to make it easier to implement, train and maintain models used in satellite image based machine learning tasks. Using different preprocessing steps it will be possible to create labels for both counting and segmentation tasks. Moreover it is made in a way that makes it easy to implement and train new models, that will be accessible through the command line interface along with possible configuration variables, that will also be exposed to the command line. Thus making it easier to train many different models with different configurations, without the need of a graphical user interface, and all accessible through a single command and a number of command line arguments.

## Installation
Clone this repository, then run 
```
pip install <path-to-repository>
```
Afterwards the package can be accessed by calling `dlc` from the command line.

## Terminology
In the following I will use the terms tile, frame and patch. These terms are relevant to the data used in https://www.nature.com/articles/s41586-020-2824-5, however similar terms should be applicable in other satellite image datasets. A tile is a satellite image. Within some tiles selected areas are annotated, meaning the trees within that area are annotated, each such area is called a frame. Since frames are not of consistent size, and some frames are very large, smaller patches of for example $256 \times 256$ pixels are fed to the model at a time, each of these are called a patch.

## Demo
The package comes in handy in 4 different steps of the pipeline, preprocessing, training, evaluation and prediction. The demo assumes that folder `/home/dataset/` has the following structure
```
dataset/
  images/
    tile1.tif
    tile2.tif
    ... // geotifs containing all the tiles
  frames/
    // empty folder to write to
  polygons.gpkg // geopandas package from which to load the
                // polygons describing the objects to count
  rectangles.gpkg // geopandas package from which to load the
                  // frames describing what areas has been annotated
```
### Preprocessing
The preprocessing step can be initiated by calling `dlc preprocess`. This takes a number of keyword arguments, the most important of which are highlighted here
```
dlc preprocess \
  --dataset_path /home/dataset/ \
  --dataset_img_path images/ \
  --output_path /home/dataset/frames/ \
  --frame_creators UniformDensityFrameCreator AltImageFrameCreator ScalarFrameDataCreator
  --count_heuristic sahel_count_heuristic
```
This will generate two images for each annotated area in `rectangles.gpkg` using the tiles in `/home/data/images/` and object annotations in `polygons.gpkg`.

The filename of the polygons and rectangles can be changed by setting `--polygons_filename` and `--rectanlges_filename` arguments respectively, these arguments take the path to the geopandas package relative to the `dataset_path`. 

`--output_path` denotes that the extracted frames will be put in `/home/dataset/frames/`

`--frame_creators` takes any number of arguments and are used to dictate what types of frames should be extracted from the tiles. All frame creators are defined in `dlc/tools/frames.py`. In the case above `AltImageFrameCreator` will simply write the image data corresponding to the frame, while `UniformDensityFrameCreator` will write an image that is 0 everywhere except in the polygons described in `polygons.gpkg`, where it will write a value such that the sum of the pixels the the polygon sums to the number of objects in the polygon (which is not necessarily one, see info on count_heuristic below). The frames will be written to `/home/dataset/frames/` and a `frames.geojson` will also be written to this location, which contains the paths to the different frames, and some scalars about each frame as computed by the `ScalarFrameDataCreator`.

`--count_heuristic` decides what count heuristic is used when counting the polygons, `sahel_count_heuristic` counts polygons with area $A < 200m^2$ as a single object, while polygons with area $A \geq 200m^2$ counts as $A \times 3 \times 10^{-2}$ objects. Currently only `sahel_count_heuristic` and `None` are available. `None` will just count a each polygon as one object.

For a full list of arguments and the default values to the preprocessing step, call `dlc preprocesss -h`

### Training
The next steps of the demo will showcase the UnetUpsamplingPath model, which is one of 3 models currently implemented. All three models take the same arguments and perform the same task, but using different architectures. The arguments and functionality shown here are specifically implemented for these three models, and in the section "Adding your own models" I will go over how to implement new models, that possibly function in a different way or with different arguments.

After the frames has been generated by the preprocessing step the training can begin. A call to the training script could look like this:
```
dlc train UnetUpsamplingPath \
  --frames_file_path /home/dataset/frames/frames.geojson \
  --n_epochs 100 \
  --base_model_file_path /home/models/pretrained_unet.h5 \ 
  --model_save_path /home/models/ \
  --input_image_keys image \
  --output_image_keys uniform-density
```
This will train the model called `UnetUpsamplingPath` for 100 epochs on the frames generated in the previous steps. 

`--base_model_file_path` is the path to the U-Net model that is used as the base model, which dense layers are appended to. `--whole_model_file_path` can be used to load in a previously trained model where the dense layers are already included, if this argument is not given the dense layers will be initialized randomly. The argument `--freeze_base_model` can be used to denote whether the U-Net should be trained or not. **Note that by default the layers of the U-Net is frozen, so if you do not have a pretrained U-Net model, this argument should explicitly be set to `False`** 

`--model_save_path` dictates the folder in which to save the model, only the model with the lowest validation loss will be saved. The filename can be changed by giving the `--model_name` argument.

`--input_image_keys` and `--output_image_keys` dictates what frames are loaded in as input images and output images/labels respectively. Here it loads the image of the given frame as input, and the uniform-density is used as output, since by summing over such a patch gives the object count in the patch, which is used as labels.

The positional argument `UnetUpsamplingPath` is what model should be trained, currently `UnetUpsamplingPath`, `UnetTopConcatLayer` and `UnetBottomLayers` are available. They all work in the same way and take the same arguments, but use different architectures. See "Adding your own models" below for information on how to implement and expose your own models to the interface.

To see all possible arguments and the default values for a specific model call `dlc train <model> -h`

### Testing/evaluating
To test a trained model the following can be used. Here it is assumed that `/home/dataset/test_frames` exists and contains preprocessed test frames.
```
dlc evaluate UnetUpsamplingPath \
  --test_frames_file_path /home/dataset/test_frames/frames.geojson \
  --whole_model_file_path /home/models/unet_upsampling_path.h5 \
  --input_image_keys image
  --output_image_keys uniform-density
```
This will evaluate the trained Unet Upsampling path model as trained before, and print the loss, accuracy and $R^2$ value.

`--test_frames_file_path` is the path to the frames to evaluate on. This argument can also be passed during training, and then after `n_epoch` epochs the best model will be loaded and evaluated on the given test set.

The evaluation of a model takes the same arguments as the training, though all the arguments may not have an effect, such as `model_name`.

### Prediction
Predictions can be performed as follows:
```
dlc predict UnetUpsamplingPath \
  --test_frames_file_path /home/dataset/test_frames/frames.geojson \
  --whole_model_file_path /home/models/unet_upsampling_path.h5 \
  --input_image_keys image
  --output_file_path /home/predictions/preds.csv
```
This will produce the count predictions of the frames in the test set and output those in a csv-file, with columns "Frame" and "Predicted count", it produces one row per patch and not per frame, so if a frame is large enough for multiple patches multiple rows will have the same frame name. The stride between patches during predictions is always equal to the patch size, meaning there is no overlap between patches. Padding is applied to the frames ensuring that the whole frame is included in the patches. Therefore the total count per frame can be found by grouping preds.csv by frame, and summing over the predicted count. 

## Adding your own models
New models can easily be implemented by following the interface of the other models. To add a model with the name YourModelName, create a folder `dlc/trainers/YourModelName` in this folder create 3 files:
```
dlc/trainers/YourModelName/
  __init__.py
  config.py
  trainer.py
```
In the config.py file implement the config class, it must be named Config, and the \_\_init\_\_ function must only take the `self` argument. All variables in the config class can be exposed to the CLI, meaning they can be altered when calling dlc by adding a keyword argument to the command line. Eg to set the vairable `var` to the value "foo" you should add the argument `--var foo` when calling the script. To expose the config variables to the CLI the variables must be defined as class variables, must not start with an underscore, and must have type annotations. The class used to parse a configuration class to an argument parser does not support every type, only the following are supported:
1. int, float, bool, str, Any type `t` that can be correctly instantiated using a string value `s` as `t(s)`. This could include custom classes.
2. List of any types from 1. (only 1 dimensional lists and lists that contain only one type are supported)
3. Tuples of any types from 1. (type annotations for each element in the tuple must be given, and nested tuples are not supported)
4. numpy arrays of any type from 1. (only 1 dimensional numpy arrays that contain only one type are supported). The type of the elements of a numpy array is dictated by the first element in the default value, if this value is `None` or the array is empty by default, type `str` is assumed.
5. Optionals of any type from 1. Giving the string value "None" to an argument with the Optional type will result in a `None` value, all other values will be tried to convert to the inner type of the optional.
6. There are no checks that the given types are supported, so errors may happen silently if non-supported types are given.

In the \_\_init\_\_ function all the class variables must be set as instance variables as well. This can for example be done using the following snippet
```python
class Config:
  # define config-vars here

  def __init__(self) -> None:
    [
        setattr(self, k, v)
        for k, v in vars(Config).items()
        if not k.startswith("_")
    ]
```
The reason for this detour for setting the config variables is that the type hints can only be fetched from the class variables, however I believe instance variables to provide a better interface for the config classes, since then the code is passing specific instances around, and not classes resulting in nicer type annotations for functions taking `Config`-objects as argument.

Config classes can inherit from each other, if that is the case, just call `super().__init__()` before setting all the instance variables in the `__init__` function. 

In trainer.py you implement the actual model and training loop. The trainer should at least implement the interface of the BaseTrainer found in dlc/trainers/base_trainer.py. If you implement a trainer for other trainers to inherit from, but not to be run on its own, that trainer should be implemented in dlc/models/, see dlc/models/UnetBasedTrainer/ for an example. This ensures that all modules in dlc/trainers/ are run-able trainers, so in a future version of the package each module in dlc/trainers/ can be listed as the options for the `trainer` argument in the command line.

The \_\_init\_\_.py you should write
```python
from .config import Config
from .trainer import Trainer
```

If done correctly you will be able to train, evaluate and predict your model with
```
dlc <{train, evaluate, predict}> YourModelName <arguments>
```

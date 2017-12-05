# Guide for Developers
## Tensorflow vs other libraries
We use tensorflow because it lets us easily deploy models on mobile devices. If you wish to move to another project for training models, you could convert trained models files to tensorflow format and use these in the android project.
## Top Level Projects
### classifier
This project contains the code for creating the model that we have used. The output of this project are in the form of pb(protobuf) files. We started this project with the hope of training on the ADE20K dataset. Eventually we have ditched our efforts to train for ADE20K directly, and started the `pretrained` subproject, which uses a pretrained network to begin with. Please visit the [ADE20K website](http://sceneparsing.csail.mit.edu/) to see what kind of data we have used.
#### Requirements
Tensorflow 1.0 or later.
Uncomment the last line in `tf_record_writer.py` and run it using `python tf_record_writer.py`. It'll download the dataset and create the `data` folder.
#### Folders/Files
* `pretrained`: contains a subproject for training ADE20K from pretrained Inception V3.
  * `pretrained/checkpoints/`*: Folder containing the checkpoint files in training sessions.
  * `pretrained/summaries/`*: Folder containing the training log files to visualize the progress using tensorboard. Change directory in this folder and run `tensorboard --logdir=./` to execute tensorboard, open `localhost:6006` in your browser to see the results (accuracy graphs over training steps and etc).
  * `pretrained/freeze.py`: Creates the `inception_v3_ade.pb` file using the checkpoint files.
  * `pretrained/model.py`: Model definition for the pretrained model.
  * `pretrained/train.py`: Trains the model defined in `model.py`.
  * `pretrained.inception_v3_pretrained.pb`: Pretrained model generated from [tensorflow models repository](https://github.com/tensorflow/models/tree/6e4bbb74beb771aa6767de867a72a3f32593f474/research/slim)
  * `pretrained/inception_v3_ade.pb`: Result of training the pretrained model and freezing it.
* `data`: [ADE20K dataset](http://sceneparsing.csail.mit.edu/). This folder is generated from `tf_record_writer.py`
  * `data/annotations`: Contains ADE20K annotations for each image.
  * `data/images`: Contains ADE20K images.
  * `data/records`: Contains the tensorflow record files for faster processing
* `test/`: Some files that we have used to experiment with tensorflow. May contain wisdom as well as nonesense.
* `ade20k.py`: Python file to access files related to the dataset.
* `blocks.py`: Some neural network block implementations for ease of use.
* `evaluate.py`: Loads the training checkpoint files and runs it against the evaluation dataset.
* `freeze.py`: Creates a frozen graph file.
* `frequencies.py`: Contains an array regarding the pixel frequencies of each class in ADE20K.
* `internal_logger.py`: A basic logging implementation to use while training.
* `pipeline.py`: reads the .tfrecord files created by `tf_record_writer.py` and creates the tensors to use in training. This is important to make your model train fast.
* `preprocess.py`: Image preprocessing functions that we use.
* `tf_record_writer.py`: Downloads and extracts the dataset, creates the tfrecord files that are going to be used by `pipeline.py`.
* `train.py`: Train the model defined in `model.py`


Files/folders that are followed by an * are not added to the git repository.

#### Training the model
Before training the model we need to set an environment in Hinton. To do so, first we need access to Hinton. Go to the groups #general channel on slack and ask who can help you to gain access to Hinton. The following steps are somewhat abstract, since I am not aiming to create an in depth guide on how to install tensorflow.
* Initial Setup
  1. Get access to Hinton (using ssh)
  2. Install [miniconda](https://conda.io/miniconda.html)
  3. Create a miniconda environment and activate it.
  3. Install CUDA and CUDNN in this environment. Search the anaconda database using this site (https://anaconda.org/search?q=cuda).
  4. Install tensorflow-gpu following the guidelines in tensorflow.org.
  5. Go to `/scratch2` and create a folder for yourself
  6. Copy the contents of this project into your folder in `/scratch2` folder. (i.e. `/scratch2/erdi/utilisation`)
  7. Run the `tf_record_writer.py` to create the tfrecord files.


* Training a job (assumes you are on hinton and activated an anaconda environment with tensorflow-gpu installed)
  1. I use `screen` to create persisting (that do not disappear when you log out) terminal sessions.
  2. Use nvidia-smi to determine if there is a GPU available.
  3. Execute your code using `CUDA_VISIBLE_DEVICES=0 python train.py` and replacing 0 with the id of the GPU device that you want to use. That id could be 0 or 1.
  4. When you run the training script, it creates the `checkpoints` and the `summaries` folders. If you want to restart a training session or make changes in your model, make sure to remove these two. Otherwise your model will may not train or continue from its last state, or your results in tensorboard will contain data from the previous training session.
  5. If you want to access tensorboard, you need to create a tunnel to the hinton using `ssh <USERNAME>@hinton.science.ru.nl -L 6006:hinton.science.ru.nl:6006 -N`. 6006 is the default port of tensorboard, replace `<USERNAME>` with your username. For example I use `ssh ecalli@hinton.science.ru.nl -L 6006:hinton.science.ru.nl:6006 -N`.
  6. Good luck!

* Making changes to the model: Depending on the change, it might involve various interventions.
  * If you want to make model level changes by adding another layer or changing the learning rate or modifying a block, you can just do these changes in the `model.py` or `blocks.py`. If you play around with the variable scopes or variable names, make sure that you reflect those changes in the freeze.py.
  * If you want to use another dataset, that would may involve creating another pipeline for input data and modifying `train.py` accordingly.
  * Using tensorflow data pipelining tools instead of feeding dictionaries to sessions increases the training performance greatly. (See https://www.tensorflow.org/apFollowing thati_guides/python/reading_data#batching)

#### The model
The latest model created from this project contains 2 output layers.
* Standard inception v3 outputs, trained for ImageNet dataset.
* 4x4 ADE20K output grid.

The layer preceding the average pooling layer of inception is a 8x8 grid with localized features. Instead of applying an average pooling layer on it, we applied average pooling on this layer, so that we have a 4x4 grid with localized features. We have applied a 1x1 convolution on this so to convert these localized features to ADE20K classes. To train this, we converted ADE20K label map to a 299x299 image using nearest neighbour algorithm. By applying a 75x75 max pooling kernel with strides of 75, we have generated an 4x4 object class grid. By applying a sigmoid cross entropy difference to the output of 1x1 convolution and the object class grid, we have calculated the loss. Training only the weights of the 1x1 convolution, we have created a good model that can tell the location of objects in an image.

### android
Files and folders in this project use a standard Android project structure. So I'm not going to give an in depth explanation of what is what.
This project has been copied from `tensorflow/examples/android`. Upon copying, we have changed the namespace of the project, tried to remove the redundant code and implemented some features on top of it.

#### Requirements
You need [android studio](https://developer.android.com/studio/index.html), android SDK (23) android build tools (26.0.1) and android NDK (ndk-bundle-14b).
* Simple: After you open this project in android studio, it should be able to run it on mobile devices immediately after configuring android studio. One important thing is preventing it from upgrading the gradle versions.
* Complicated, build from sources:
  1. Clone tensorflow using `git clone --recurse-submodules https://github.com/tensorflow/tensorflow.git`
  2. Move the root project folder `utilisation` inside `tensorflow/tensorflow`, such that when you change directory to the tensorflow root folder, you can access the `android` project folder using relative path `tensorflow/utilisation/android`. Please note that there is a folder called `tensorflow` inside the `tensorflow` root folder.
  3. Edit the WORKSPACE file so that it has the proper android_sdk_repository and android_sdk_repository configurations. In my case I have these settings.
  ```
  android_sdk_repository(
    name = "androidsdk",
    api_level = 23,
    # Ensure that you have the build_tools_version below installed in the
    # SDK manager as it updates periodically.
    build_tools_version = "26.0.1",
    # Replace with path to Android SDK on your system
    path = "/home/erdi/Android/Sdk",
  )
  android_ndk_repository(
      name="androidndk",
      path="/home/erdi/Android/Sdk/ndk-bundle-14b",
      # This needs to be 14 or higher to compile TensorFlow.
      # Please specify API level to >= 21 to build for 64-bit
      # archtectures or the Android NDK will automatically select biggest
      # API level that it supports without notice.
      # Note that the NDK version is not the API level.
      api_level=14)
  ```
  4. Install bazel
  5. Build the project using `bazel build -c opt //tensorflow/utilisation/android:tensorflow_demo`
  6. To install it on your mobile phone, use `adb install -r bazel-bin/tensorflow/utilisation/android/tensorflow_demo.apk`
  7. See the `README.md` file in [Tensorflow android example](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/android) or the `README.md` file in `android` project.


#### How it works?
We have generated a model file using the `classifier` project. The android project assumes that this model file is copied to the `assets` folder. This project uses the Android NDK (Native Development Kit) to load the compiled Tensorflow C API (`lib_tensorflow.so`). Implements and imports some JNI classes to interact with this API using the android (Java) sources. Upon request, java sources load the model file and create a tensorflow session with that. Using the JNI we feed data to the session, run it and get outputs in Java (See: `android/src/net/ccnlab/eyecontact/TensorFlowImageClassifier.java`).


#### Features
* Current model supports a set of localized results for the ADE20K classes. We also have the corresponding interface implementations in the git history. However we have removed these from the android project because in the meeting with our users we have seen that such a representation (or the way we implemented the interface) was not very helpful to them.
* `assets/class_tree.txt` is parsed to create the class selection categories. This file is created by the combination of my personal input and imagenet competition synets. See project `class_tree` for more information. If you want to make changes in the ordering or the hierarchy of these categories, feel free to edit this file. Selectable classes in this file contain a class id preceded by a `:`. For example `library:625` in line 14. It means that the class library is associated with the array index `625` of the output layer of the model.
* We have implemented voice commands api of google, so that a user can make selections using speech. However it does not work as good as we thought it would. First, to use this feature, one needs to trigger the application from google assistant by saying "take a picture with eyecontact". Then the user needs to tell the name of the class or category that they want to select.

### class_tree
We have created this small project to be able to convert imagenet synets to a set of categories. This was supposed to run only once to generate an initial class categorization. After that we have manually clustered those classes so that it's easy for our users to find classes.

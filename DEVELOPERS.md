
# Working on Hinton
Before training models we need to set an environment in Hinton. To do so, first we need access to Hinton. Go to the groups #general channel on slack and ask who can help you to gain access to Hinton. The following steps are somewhat abstract, since I am not aiming to create an in depth guide on how to install tensorflow.
* Initial Setup
  1. Get access to Hinton (using ssh)
  2. Install [miniconda](https://conda.io/miniconda.html)
  3. Create a miniconda environment and activate it.
  3. Install CUDA and CUDNN in this environment. Search the anaconda database using this site (https://anaconda.org/search?q=cuda).
  4. Install tensorflow-gpu following the guidelines in tensorflow.org.
  5. Go to `/scratch2` and create a folder for yourself
  6. Copy the contents of this project into your folder in `/scratch2` folder. (i.e. `/scratch2/erdi/utilisation`)

* Training a model (assumes you are on hinton and activated an anaconda environment with tensorflow-gpu installed)
  1. I use `screen` to create persisting (so that your jobs do not disappear when you log out) terminal sessions.
  2. Use nvidia-smi to determine if there is a GPU available.
  3. Execute your code using `CUDA_VISIBLE_DEVICES=0 python script.py` and replacing `0` with the id of the GPU device that you want to use (could be `0` and `1`) and `script.py` with the script that you want to run.
  4. When you run the training script, it creates the `checkpoints` and the `summaries` folders. If you want to restart a training session or make changes in your model, make sure to remove these two. Otherwise your model will may not train or continue from its last state, or your results in tensorboard will contain data from the previous training session.
  5. If you want to access tensorboard, you need to create a tunnel to the hinton using `ssh <USERNAME>@hinton.science.ru.nl -L 6006:hinton.science.ru.nl:6006 -N`. 6006 is the default port of tensorboard, replace `<USERNAME>` with your username. For example I use `ssh ecalli@hinton.science.ru.nl -L 6006:hinton.science.ru.nl:6006 -N`.
  6. Good luck!

# The model
The latest model created from this project can be seen in [readme file of image_retraining](image_retraining/README.md)

# Android
There are 2 android folders, `android_lite` and `android_shared`. `android_shared` contains the shared codebase and `android_lite` extends that project. (At some point there were two android projects that were using this shared codebase)
Files and folders in this project use a standard Android project structure. So I'm not going to give an in depth explanation of what is what.
This project has been copied from `tensorflow/examples/android`. Upon copying, we have changed the namespace of the project, tried to remove the redundant code and implemented some features on top of it.

### Requirements
You need [android studio](https://developer.android.com/studio/index.html), android SDK (23) android build tools (26.0.1) and android NDK (ndk-bundle-14b). *It works best on ubuntu*.
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


### How it works?
We have generated a model file using the `image_retraining` project. The android project assumes that this model file is copied to the `assets` folder and referred in the `assets/class_tree.txt` file. This project uses the Android NDK (Native Development Kit) to load the compiled Tensorflow C API (`lib_tensorflow.so`). Implements and imports some JNI classes to interact with this API using the android (Java) sources. Upon request, java sources load the model file and create a tensorflow session with that. Using the JNI we feed data to the session, run it and get outputs in Java (See: `android/src/net/ccnlab/eyecontact/TensorFlowImageClassifier.java`).

### Next Steps
* We may need to add more classes.
* When there are multiple object types (a wallet and a key) in the image, we fail to claim that either is there with high confidence.
* pitch tone change can be supported by vibration.
* Instead of pitch tone change, we could implement something like a "geiger counter" which constantly clicks but reduces the interval between clicks based on the confidence.
* when the phone is parallel to the ground, the model works best. In cases where people are looking ahead, it doesn't work that well. We could try more data augmentations (image transformations and rotations) to fix that.

### Important commits
 * ADE20K "localized outputs" are removed completely with commit id `534f8d8901086923c71fabe937938c38c741adc5`.
 * ADE20K "localized outputs" are removed from the android app in commit `0faf138aca9e257c1d7bcd0c41fe45fb14d59457`.
 * `android_selector` is removed with commit id `534f8d8901086923c71fabe937938c38c741adc5`.
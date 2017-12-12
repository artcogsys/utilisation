# Extending the Classifier
This document is meant to explain the steps and the possible ways implement them to extend the current model with new object classes. I assume (and advise) that you will be using the pretrained weights in the current model. This document was written for Tensorflow version 1.4, some function names may change in future versions.

## 1. Data
If you want to add a new class to the classifier, this step is explaining how you can do that. Once you know the class that you want to add to the model, you can check the full ImageNet dataset to see if it contains that class.

ImageNet is extensively tagged with various types of objects and categorized using the extensive wordnet hierarchy. For potential classes that you can add to the classifier, see [the ImageNet tree view]](http://image-net.org/explore).

To download this data, you need to
1. [Sign Up to Image Net](http://image-net.org/signup)
2. [Log In](http://image-net.org/login)
3. [Go to the ImageNet tree view](http://image-net.org/explore)
4. Find the class(es) that you want to add.
5. Click on *Downloads* above the images.
6. Use the link under *Download images in the synset*.

You may also want to use a different dataset with different classes.

## 2. Data Pipeline
To train the model using this new data, you need to create a pipeline that can read it. To achieve this, you can follow the [tensorflow guidelines](https://www.tensorflow.org/api_guides/python/reading_data#batching). This guidenline walks you through setting up a set of tensors that provide different batches in every session execution.

Another way to achieve this is to read those files using python libraries outside tensorflow and providing batches for placeholder variables through the `feed_dict` argument of `session.run`. See the [tensorflow guidelines](https://www.tensorflow.org/api_guides/python/reading_data#Feeding) for more information.

## 3. Defining the Model
One very easy way to extend a model is to read the previous model definition from the frozen graph. We have used (and probably invented) this method in `classifier/pretrained/model.py`. This methods works by,
1. Create a graph object using `tf.Graph()`. Create a `graph.as_default` block and execute the following steps within it.
2. Adding the input tensors defined in the Data Pipeline section.
3. Importing the `.pb` file that contains the frozen graph definition into this session by using `tf.import_graph_def`.
4. Setting the input tensors defined in step 2 as the input of the imported graph using the `input_map` argument of `tf.import_graph_def`.
5. Getting the desired output tensors of the imported model from the graph definition using `graph.get_tensor_by_name`. To see the available tensors and operations, you can use `graph.get_operations`. Note that a tensor refers to the output of an operation. Therefore, if you want to get the tensor that is the output of an operation, you need to add `:0` prefix to its name. For example, an operation named as `operation` will have an output tensor named `operation:0`.
6. Implement the layer(s) that you wish following these tensors.
7. Create a loss tensor (e.g. `tf.nn.sigmoid_cross_entropy_with_logits`, `tf.nn.softmax_cross_entropy_with_logits`).
8. Create an optimizer (e.g. `tf.train.GradientDescentOptimizer`).
9. Determine the weight variables that you want to train.
    * I recommend training only the weight variables that you have defined in step 6. This is useful when you are adding new classes to the existing model.
    * You can also train the whole network. However, if you do that, you must also retrain (or remove) the original outputs (1000 classes from ImageNet and 150 classes from ADE20K) of the model.
10. Create a training step only for the weight variables that you want to train for. To do that, you should use the `var_list` argument of the `minimize` function of the `optimizer` that you have defined in step 8.

## 4. Training the Model
Import the graph you have defined into a session and evaluate the training step. Create a `tf.train.Saver` for the variables that you are training and save a checkpoint file at the end.

This is standard for training tensorflow models, so I'm not going into details here.

## 5. Freezing the Model
Import the graph that you have defined in section 3 again. However, this time when you are setting the `input_map` argument of `tf.import_graph_def`, provide placeholder tensor(s). Import the whole graph into a session, load the trained weights from  the checkpoint file. Use `tf.graph_util.convert_variables_to_constants` to create a graph_def with our variables converted to constants, also provide the names of output tensors that you desire. Be careful here, if you only use the operations, that you have appended to the previous graph, you may break existing functionality in the Android app. Write the output of `graph_def.SerializeToString()` to a file.

## 6. Importing the model to the Android project
Android project uses the `utilisation/android/assets` folder to read the pb file. Place the new model in that folder, remove the old one, make related changes in the `TensorFlowImageClassifier.java`.

----
All these steps, except using new classes from ImageNet, are used in the `pretrained` folder to train a model on ADE20K. See the code if you have questions or reach me at [my email address](mailto:erdicalli\(AT\)gmail.com).

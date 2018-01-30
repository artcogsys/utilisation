/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

package net.ccnlab.eyecontact;

import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.os.Trace;

import net.ccnlab.eyecontact.model.ClassificationResult;

import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

import java.io.IOException;

/**
 * A classifier specialized to label images using TensorFlow.
 */
public class TensorFlowImageClassifier implements Classifier {
    private static final String TAG = "ImageClassifier";


    private static final int INPUT_SIZE = 299;
    private static final int IMAGE_MEAN = 128;
    private static final float IMAGE_STD = 128;
    private static final String INPUT_NAME = "Placeholder";

    private static final String CLASSIFICATION_OUTPUT_NAME = "aggregated_class_output/Sum";

    private static final String MODEL_FILE = "file:///android_asset/inception_v3_ade.pb";


    private int[] classIdsToFind;
    private String classLabel;
    // Config values.
    private String inputName;
    private String classificationOutputName;
    private int inputSize;
    private int imageMean;
    private float imageStd;

    // Pre-allocated buffers.
    private int[] intValues;
    private float[] floatValues;
    private float[] classificationOutput;

    private String[] outputNames;

    private boolean logStats = false;

    private TensorFlowInferenceInterface inferenceInterface;

    private TensorFlowImageClassifier() {
    }

    /**
     * Initializes a native TensorFlow session for classifying images.
     *
     * @param assetManager             The asset manager to be used to load assets.
     * @param modelFilename            The filepath of the model GraphDef protocol buffer.
     * @throws IOException
     */
    public static Classifier create(
            AssetManager assetManager,
            String modelFilename,
            int[] classIdsToFind,
            String classLabel) {
        TensorFlowImageClassifier c = new TensorFlowImageClassifier();
        c.inputName = INPUT_NAME;
        c.classificationOutputName = CLASSIFICATION_OUTPUT_NAME;

        c.inferenceInterface = new TensorFlowInferenceInterface(assetManager, MODEL_FILE);

        c.inputSize = INPUT_SIZE;
        c.imageMean = IMAGE_MEAN;
        c.imageStd = IMAGE_STD;

        // Pre-allocate buffers.
        c.outputNames = new String[]{c.classificationOutputName};
        c.intValues = new int[INPUT_SIZE*INPUT_SIZE];
        c.floatValues = new float[INPUT_SIZE*INPUT_SIZE* 3];
        c.classIdsToFind = classIdsToFind;
        c.classLabel = classLabel;
        c.classificationOutput = new float[1];
        return c;
    }

    @Override
    public ClassificationResult recognizeImage(final Bitmap bitmap) {
        // Log this method so that it can be analyzed with systrace.
        Trace.beginSection("recognizeImage");

        Trace.beginSection("preprocessBitmap");
        // Preprocess the image data from 0-255 int to normalized float based
        // on the provided parameters.
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        for (int i = 0; i < intValues.length; ++i) {
            final int val = intValues[i];
            floatValues[i * 3] = (((val >> 16) & 0xFF) - imageMean) / imageStd;
            floatValues[i * 3 + 1] = (((val >> 8) & 0xFF) - imageMean) / imageStd;
            floatValues[i * 3 + 2] = ((val & 0xFF) - imageMean) / imageStd;
        }
        Trace.endSection();

        // Copy the input data into TensorFlow.
        Trace.beginSection("feed");
        inferenceInterface.feed(inputName, floatValues, 1, inputSize, inputSize, 3);
        inferenceInterface.feed("aggregated_class_output/Placeholder", classIdsToFind, classIdsToFind.length);
        Trace.endSection();

        // Run the inference call.
        Trace.beginSection("run");
        inferenceInterface.run(outputNames, logStats);
        Trace.endSection();

        // Copy the output Tensor back into the output array.
        Trace.beginSection("fetch");
        inferenceInterface.fetch(classificationOutputName, classificationOutput);
        Trace.endSection();

        Trace.beginSection("setResults");

        Trace.endSection();

        Trace.endSection(); // "recognizeImage"
        return new ClassificationResult(classLabel, classificationOutput[0]);
    }

    @Override
    public void enableStatLogging(boolean logStats) {
        this.logStats = logStats;
    }

    @Override
    public String getStatString() {
        return inferenceInterface.getStatString();
    }

    @Override
    public void close() {
        inferenceInterface.close();
    }

    public int getInputSize() {
        return inputSize;
    }
}

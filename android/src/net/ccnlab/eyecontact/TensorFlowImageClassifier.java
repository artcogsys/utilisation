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
import android.util.Log;

import net.ccnlab.eyecontact.model.ResultsContainer;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Vector;

import org.tensorflow.Operation;
import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

/**
 * A classifier specialized to label images using TensorFlow.
 */
public class TensorFlowImageClassifier implements Classifier {
    private static final String TAG = "ImageClassifier";

    // Only return this many results with at least this confidence.
    private static final float THRESHOLD = 0.6f;

    // Config values.
    private String inputName;
    private String localizedLabelOutputName;
    private String classificationOutputName;
    private int inputSize;
    private int imageMean;
    private float imageStd;

    // Pre-allocated buffers.
    private Vector<String> localizedLabelNames = new Vector<String>();
    private Vector<String> classificationLabelNames = new Vector<String>();
    private int[] intValues;
    private float[] floatValues;
    private float[] localizedLabelOutputs;
    private float[] classificationOutputs;

    private String[] outputNames;

    private boolean logStats = false;

    private TensorFlowInferenceInterface inferenceInterface;

    private TensorFlowImageClassifier() {
    }

    /**
     * Initializes a native TensorFlow session for classifying images.
     *
     * @param assetManager  The asset manager to be used to load assets.
     * @param modelFilename The filepath of the model GraphDef protocol buffer.
     * @param classificationLabelNameFilename The filepath of label file for classes.
     * @param localizedLabelNameFilename The filepath of label file for localized labels.
     * @param inputSize     The input size. A square image of inputSize x inputSize is assumed.
     * @param imageMean     The assumed mean of the image values.
     * @param imageStd      The assumed std of the image values.
     * @param inputName     The label of the image input node.
     * @param classificationOutputName    The label of the classification output node.
     * @param localizedLabelOutputName    The label of the localized label output node.
     * @throws IOException
     */
    public static Classifier create(
            AssetManager assetManager,
            String modelFilename,
            String classificationLabelNameFilename,
            String localizedLabelNameFilename,
            int inputSize,
            int imageMean,
            float imageStd,
            String inputName,
            String classificationOutputName,
            String localizedLabelOutputName) {
        TensorFlowImageClassifier c = new TensorFlowImageClassifier();
        c.inputName = inputName;
        c.localizedLabelOutputName = localizedLabelOutputName;
        c.classificationOutputName = classificationOutputName;

        setLabels(classificationLabelNameFilename, c.classificationLabelNames, assetManager);
        setLabels(localizedLabelNameFilename, c.localizedLabelNames, assetManager);

        c.inferenceInterface = new TensorFlowInferenceInterface(assetManager, modelFilename);

        c.localizedLabelOutputs = getOutputBuffer(localizedLabelOutputName, c.inferenceInterface, 4);
        c.classificationOutputs = getOutputBuffer(classificationOutputName, c.inferenceInterface, 2);

        c.inputSize = inputSize;
        c.imageMean = imageMean;
        c.imageStd = imageStd;

        // Pre-allocate buffers.
        c.outputNames = new String[]{localizedLabelOutputName, classificationOutputName};
        c.intValues = new int[inputSize * inputSize];
        c.floatValues = new float[inputSize * inputSize * 3];

        return c;
    }

    @Override
    public ResultsContainer recognizeImage(final Bitmap bitmap) {
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
        Trace.endSection();

        // Run the inference call.
        Trace.beginSection("run");
        inferenceInterface.run(outputNames, logStats);
        Trace.endSection();

        // Copy the output Tensor back into the output array.
        Trace.beginSection("fetch");
        inferenceInterface.fetch(localizedLabelOutputName, localizedLabelOutputs);
        inferenceInterface.fetch(classificationOutputName, classificationOutputs);
        Trace.endSection();

        Trace.beginSection("setResults");

        ResultsContainer resultsContainer = new ResultsContainer(classificationLabelNames, localizedLabelNames);
        resultsContainer.setClassificationResults(classificationOutputs);
        resultsContainer.setLocalizedLabelResults(localizedLabelOutputs);

        Trace.endSection();


        Trace.endSection(); // "recognizeImage"
        return resultsContainer;
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

    private static void setLabels(String filename, Vector<String> buffer, AssetManager assetManager) {
        String actualFilename = filename.split("file:///android_asset/")[1];
        Log.i(TAG, "Reading label names from: " + actualFilename);
        BufferedReader br = null;
        try {
            br = new BufferedReader(new InputStreamReader(assetManager.open(actualFilename)));
            String line;
            while ((line = br.readLine()) != null) {
                buffer.add(line);
            }
            br.close();
        } catch (IOException e) {
            throw new RuntimeException("Problem reading label file!", e);
        }
    }

    // Does not support partially known dimensions.
    private static float[] getOutputBuffer(String operationName, TensorFlowInferenceInterface inferenceInterface, int dimensionCount) {
        Log.i(TAG, "creating output buffers for: " + operationName);

        final Operation operation = inferenceInterface.graphOperation(operationName);
        Log.i(TAG, "got operation for name: " + operationName);
        int shape = 1;
        int dim = 0;
        while (dim < dimensionCount) {
            shape *= (int) operation.output(0).shape().size(dim);
            dim++;
        }

        return new float[shape];
    }
}

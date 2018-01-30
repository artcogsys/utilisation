/*
 * Copyright 2016 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package net.ccnlab.eyecontact;

import android.graphics.Bitmap;
import android.graphics.Bitmap.Config;
import android.graphics.Canvas;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Typeface;
import android.media.ImageReader.OnImageAvailableListener;
import android.os.SystemClock;
import android.util.Size;
import android.util.TypedValue;
import android.view.Display;

import java.io.IOException;
import java.util.Vector;

import net.ccnlab.eyecontact.env.BorderedText;
import net.ccnlab.eyecontact.env.ImageUtils;
import net.ccnlab.eyecontact.env.Logger;
import net.ccnlab.eyecontact.model.ClassificationResult;

public class ClassifierActivity extends CameraActivity implements OnImageAvailableListener {
    private static final Logger LOGGER = new Logger();

    protected static final boolean SAVE_PREVIEW_BITMAP = false;

    private ClassificationResultView classificationResultView;

    private Bitmap rgbFrameBitmap = null;
    private Bitmap croppedBitmap = null;

    private static final boolean MAINTAIN_ASPECT = true;

    private static final Size DESIRED_PREVIEW_SIZE = new Size(640, 480);

    private Integer sensorOrientation;
    private Classifier classifier;
    private Matrix frameToCropTransform;
    private Matrix cropToFrameTransform;

    private BorderedText borderedText;

    @Override
    protected int getLayoutId() {
        return R.layout.camera_connection_fragment;
    }

    @Override
    protected Size getDesiredPreviewFrameSize() {
        return DESIRED_PREVIEW_SIZE;
    }

    private static final float TEXT_SIZE_DIP = 10;

    @Override
    public void onPreviewSizeChosen(final Size size, final int rotation) {

        final float textSizePx = TypedValue.applyDimension(
                TypedValue.COMPLEX_UNIT_DIP, TEXT_SIZE_DIP, getResources().getDisplayMetrics());
        borderedText = new BorderedText(textSizePx);
        borderedText.setTypeface(Typeface.MONOSPACE);
        LOGGER.i("creating the classifier");
        int[] classIdsToFind = getIntent().getIntArrayExtra("classIds");

        String classLabel = getIntent().getStringExtra("classLabel");
        String modelFile = getIntent().getStringExtra("modelFile");

        classifier =
                TensorFlowImageClassifier.create(
                        getAssets(),
                        modelFile,
                        classIdsToFind,
                        classLabel);
        LOGGER.i("created the classifier");
        previewWidth = size.getWidth();
        previewHeight = size.getHeight();
        LOGGER.i("getting display the classifier");
        final Display display = getWindowManager().getDefaultDisplay();
        final int screenOrientation = display.getRotation();

        LOGGER.i("Sensor orientation: %d, Screen orientation: %d", rotation, screenOrientation);

        sensorOrientation = rotation + screenOrientation;

        LOGGER.i("Initializing at size %dx%d", previewWidth, previewHeight);
        rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Config.ARGB_8888);
        croppedBitmap = Bitmap.createBitmap(classifier.getInputSize(), classifier.getInputSize(), Config.ARGB_8888);

        frameToCropTransform = ImageUtils.getTransformationMatrix(
                previewWidth, previewHeight,
                classifier.getInputSize(), classifier.getInputSize(),
                sensorOrientation, MAINTAIN_ASPECT);

        cropToFrameTransform = new Matrix();
        frameToCropTransform.invert(cropToFrameTransform);


    }

    @Override
    protected void processImage() {
        rgbFrameBitmap.setPixels(getRgbBytes(), 0, previewWidth, 0, 0, previewWidth, previewHeight);
        final Canvas canvas = new Canvas(croppedBitmap);
        canvas.drawBitmap(rgbFrameBitmap, frameToCropTransform, null);

        if (SAVE_PREVIEW_BITMAP) {
            ImageUtils.saveBitmap(croppedBitmap);
        }
        runInBackground(
                new Runnable() {
                    @Override
                    public void run() {
                        final ClassificationResult results = classifier.recognizeImage(croppedBitmap);
                        LOGGER.i("Detect: %s", results);

                        if (classificationResultView == null) {
                            classificationResultView = (ClassificationResultView) findViewById(R.id.classification_result_view);
                        }
                        classificationResultView.setResults(results);
                        readyForNextImage();
                    }
                });
    }

    @Override
    public synchronized void onStop() {
        super.onStop();
        classificationResultView.stopTune();
    }

    @Override
    public synchronized void onPause() {
        super.onPause();
        classificationResultView.stopTune();
    }
}

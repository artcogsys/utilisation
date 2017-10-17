/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Paint;
import android.util.AttributeSet;
import android.util.TypedValue;
import android.view.View;
import android.view.ViewGroup;
import android.widget.GridView;

import net.ccnlab.eyecontact.env.Logger;

import java.util.List;

public class RecognitionScoreView extends GridView implements ResultsView {
    private static final float TEXT_SIZE_DIP = 10;
    private final float textSizePx;
    private final Paint fgPaint;
    private final Paint bgPaint;
    private GridViewAdapter gridViewAdapter = null;

    public RecognitionScoreView(final Context context, final AttributeSet set) {
        super(context, set);
        textSizePx =
                TypedValue.applyDimension(
                        TypedValue.COMPLEX_UNIT_DIP, TEXT_SIZE_DIP, getResources().getDisplayMetrics());
        fgPaint = new Paint();
        fgPaint.setTextSize(textSizePx);

        bgPaint = new Paint();
        bgPaint.setColor(0xcc4285f4);
//        this.getParent().
//        View texture = findViewById(R.id.texture);
    }

    public GridViewAdapter getGridViewAdapter() {
        if (gridViewAdapter == null) {
            gridViewAdapter = new GridViewAdapter(getContext());
            this.setAdapter(gridViewAdapter);
        }
        return gridViewAdapter;
    }

    @Override
    public void setResults(final List<Classifier.Recognition> results) {
        getGridViewAdapter().setResults(results);
        postInvalidate();
    }

    @Override
    public void onDraw(final Canvas canvas) {
        canvas.drawPaint(bgPaint);
        getGridViewAdapter().updateResults();
    }
    public void setShape(int width, int height) {
        updateViewLayout(this, new ViewGroup.LayoutParams(width, height));
    }

}

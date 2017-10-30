package net.ccnlab.eyecontact;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Paint;
import android.util.AttributeSet;
import android.util.TypedValue;
import android.widget.TextView;


import net.ccnlab.eyecontact.env.Logger;
import net.ccnlab.eyecontact.model.ClassificationResult;

import java.util.List;

public class ClassificationResultView extends TextView implements ResultsView<ClassificationResult> {
    private static final Logger LOGGER = new Logger();

    private static final float TEXT_SIZE_DIP = 15;
    private List<ClassificationResult> results;

    private final StringBuilder sb = new StringBuilder();

    public ClassificationResultView(final Context context, final AttributeSet set) {
        super(context, set);
        setBackgroundColor(0xFF009ec8);
        setTextColor(0xFF000000);
    }

    @Override
    public void setResults(final List<ClassificationResult> results) {
        this.results = results;
        postInvalidate();
    }

    @Override
    public void onDraw(final Canvas canvas) {
        sb.setLength(0); // set length of buffer to 0
        sb.trimToSize();

        if (results != null) {
            for (final ClassificationResult recog : results) {
                sb.append(recog.getTitle()).append('\n');
                LOGGER.i(recog.getTitle());
            }

            setText(sb.toString());
        }
    }
}

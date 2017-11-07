package net.ccnlab.eyecontact;

import android.content.Context;
import android.graphics.Typeface;
import android.os.Handler;
import android.util.AttributeSet;
import android.widget.TextView;

import net.ccnlab.eyecontact.env.Logger;
import net.ccnlab.eyecontact.model.ClassificationResult;

import java.util.List;

public class ClassificationResultView extends AccessibilityUpdatingTextView implements ResultsView<ClassificationResult> {

    private final Logger LOGGER = new Logger();
    private final StringBuilder sb = new StringBuilder();
    private Handler handler;
    public ClassificationResultView(final Context context, final AttributeSet set) {
        super(context, set);
        Typeface tf = Typeface.createFromAsset(context.getAssets(),
                "calibri.ttf");
        setBackgroundColor(0xFF009ec8);
        setTextColor(0xFF000000);
        setTypeface(tf);
        handler = new Handler();
    }

    @Override
    public void setResults(final List<ClassificationResult> results) {
        sb.setLength(0);
        sb.trimToSize();

        if (results != null) {
            for (final ClassificationResult recog : results) {
                sb.append(recog.getTitle()).append('\n');
                LOGGER.i("%s: %.2f", recog.getTitle(), recog.getConfidence());
            }
            handler.post(new Runnable() {
                @Override
                public void run() {
                    setText(sb.toString());
                }
            });
        }
    }
}

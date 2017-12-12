package net.ccnlab.eyecontact;

import android.content.Context;
import android.graphics.Typeface;
import android.os.Handler;
import android.util.AttributeSet;
import android.widget.TextView;

import net.ccnlab.eyecontact.env.Logger;
import net.ccnlab.eyecontact.model.ClassificationResult;

import java.util.List;
import java.util.Locale;

public class ClassificationResultView extends AccessibilityUpdatingTextView implements ResultsView<ClassificationResult> {

    private final Logger LOGGER = new Logger();
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
    public void setResults(final ClassificationResult result) {
        if (result != null) {
            LOGGER.i("%s: %.2f", result.getTitle(), result.getConfidence());
            final String resultString = String.format(Locale.ENGLISH,"%.2f", result.getConfidence());
            handler.post(new Runnable() {
                @Override
                public void run() {
                    setText(resultString);
                }
            });
        }
    }
}

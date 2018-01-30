package net.ccnlab.eyecontact;

import android.content.Context;
import android.graphics.Typeface;
import android.os.Handler;
import android.util.AttributeSet;
import android.view.accessibility.AccessibilityEvent;

import com.karlotoy.perfectune.instance.PerfectTune;

import net.ccnlab.eyecontact.env.Logger;
import net.ccnlab.eyecontact.model.ClassificationResult;

import java.util.Locale;

public class ClassificationResultView extends AccessibilityUpdatingTextView implements ResultsView<ClassificationResult> {

    private static final PerfectTune perfectTune = new PerfectTune();
    private final Logger LOGGER = new Logger();
    private final double[] TUNES = {440.0, 493.88, 523.25, 587.33, 659.25, 698.46, 783.99, 880.00, 987.77, 1046.50, 1174.66};

    private Handler handler;

    public ClassificationResultView(final Context context, final AttributeSet set) {
        super(context, set);
        Typeface tf = Typeface.createFromAsset(context.getAssets(),
                "calibri.ttf");
        setBackgroundColor(0xFF009ec8);
        setTextColor(0xFF000000);
        setTextSize(24);
        setTypeface(tf);
        handler = new Handler();
    }

    @Override
    public void setResults(final ClassificationResult result) {
        if (result != null) {
            perfectTune.playTune();
            LOGGER.i("%s: %.2f", result.getTitle(), result.getConfidence());
            int scaledConfidence = Math.round(result.getConfidence() * 10);
            final String resultString = String.format(Locale.ENGLISH, "%d", scaledConfidence);
            perfectTune.setTuneFreq(TUNES[scaledConfidence]);

            handler.post(new Runnable() {
                @Override
                public void run() {
                    setText(resultString);
                    requestFocus();
                    sendAccessibilityEvent(AccessibilityEvent.TYPE_VIEW_FOCUSED);
                }
            });

        }
    }

    public void stopTune() {
        perfectTune.stopTune();
    }


}

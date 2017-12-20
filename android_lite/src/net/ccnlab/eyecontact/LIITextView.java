package net.ccnlab.eyecontact;

import android.content.Context;
import android.graphics.Typeface;
import android.util.AttributeSet;
import android.widget.TextView;


public class LIITextView extends TextView {
    public LIITextView(Context context, AttributeSet attrs) {
        super(context, attrs);
        Typeface tf = Typeface.createFromAsset(context.getAssets(),
                "calibri.ttf");
        setTypeface(tf);
    }
}

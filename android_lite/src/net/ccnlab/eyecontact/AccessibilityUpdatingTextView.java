package net.ccnlab.eyecontact;

import android.content.Context;
import android.graphics.Typeface;
import android.os.Bundle;
import android.util.AttributeSet;
import android.view.View;
import android.view.ViewGroup;
import android.view.accessibility.AccessibilityEvent;
import android.view.accessibility.AccessibilityNodeInfo;
import android.view.accessibility.AccessibilityNodeProvider;
import android.widget.TextView;

import net.ccnlab.eyecontact.env.Logger;

public class AccessibilityUpdatingTextView extends TextView {

    private long latestAnnouncementTime;
    private Logger LOGGER = new Logger();
    public AccessibilityUpdatingTextView(Context context) {
        this(context, null);
    }

    public AccessibilityUpdatingTextView(final Context context, final AttributeSet set) {
        super(context, set);
        setAccessibilityDelegate(new View.AccessibilityDelegate() {
            @Override
            public void onInitializeAccessibilityNodeInfo(View host, AccessibilityNodeInfo info) {
                // disables the "double click to activate" saying
                super.onInitializeAccessibilityNodeInfo(host, info);
            }
        });
        setClickable(false);
        Typeface tf = Typeface.createFromAsset(context.getAssets(),
                "calibri.ttf");
        setTypeface(tf);
        setTextSize(15);
    }

    @Override
    public void setText(CharSequence text, BufferType type) {
        super.setText(text, type);
        if (isAccessibilityFocused()) {
            long now = System.currentTimeMillis();
//            LOGGER.d("ANNOUNCEMENT %d", (now - latestAnnouncementTime));
//            boolean returnValue;
            if (now - latestAnnouncementTime > 2500) {
                LOGGER.d("FIRED ANNOUNCEMENT %d", now - latestAnnouncementTime);
                latestAnnouncementTime = now;
                announceForAccessibility(text);
            } else {
                LOGGER.d("SKIPPING ANNOUNCEMENT %d", now - latestAnnouncementTime);
            }
        }
    }

}

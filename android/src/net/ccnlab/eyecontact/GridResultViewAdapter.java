package net.ccnlab.eyecontact;

import android.content.Context;
import android.graphics.Typeface;
import android.view.View;
import android.view.ViewGroup;
import android.view.accessibility.AccessibilityEvent;
import android.view.accessibility.AccessibilityNodeInfo;
import android.widget.BaseAdapter;
import android.widget.TextView;

import net.ccnlab.eyecontact.env.Logger;
import net.ccnlab.eyecontact.model.LocalizedLabel;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

public class GridResultViewAdapter extends BaseAdapter {
    private List<LocalizedLabel> results;
    private List<List<LocalizedLabel>> resultsByGrid;
    private final StringBuilder sb;
    private List<TextView> textViews;
    private boolean clean = false;
    private static final int cellBackgroundColor = 0x99009ec8;
    private final Logger LOGGER = new Logger();

    GridResultViewAdapter(Context c) {
        super();

        this.results = new ArrayList<>();
        this.resultsByGrid = new ArrayList<>();
        textViews = new ArrayList<>();
        for (int i = 0; i < 16; i++) {
            resultsByGrid.add(i, new ArrayList<LocalizedLabel>());
            TextView textView = new AccessibilityUpdatingTextView(c);
            textViews.add(textView);
        }
        this.sb = new StringBuilder();
    }

    @Override
    public int getCount() {
        return 16;
    }

    @Override
    public Object getItem(int i) {
        return results.get(i);
    }

    @Override
    public long getItemId(int i) {
        return i / 4;
    }

    @Override
    public View getView(int i, View view, ViewGroup viewGroup) {
        if (!clean) {
            viewGroup.removeAllViewsInLayout();
            clean = true;
        }
        TextView textView = textViews.get(i);

        textView.setBackgroundColor(cellBackgroundColor);
        textView.setTextColor(0xFF000000);
        textView.setHeight(viewGroup.getHeight() / 4);
        textView.setWidth(viewGroup.getWidth() / 4);
        return textView;
    }

    public List<LocalizedLabel> getResults() {
        return results;
    }

    public void setResults(List<LocalizedLabel> results) {
        this.results = results;
    }

    void updateResults() {
        int last_index = 0;
        int last_grid = -1;
        for (int i = 0; i < 16; i++) {
            // when we se each grid the first time, clean it first.
            if (last_grid < i) {
                this.resultsByGrid.get(i).clear();
                last_grid = i;
            }

            while (last_index < results.size() && results.get(last_index).getGridLocation() == i) {
                this.resultsByGrid.get(i).add(results.get(last_index));
                last_index++;
            }

            Collections.sort(this.resultsByGrid.get(i), new Comparator<LocalizedLabel>() {
                @Override
                public int compare(LocalizedLabel lrec, LocalizedLabel rrec) {
                    return Float.compare(rrec.getConfidence(), lrec.getConfidence());
                }
            });
            textViews.get(i).setText(resultsToString(this.resultsByGrid.get(i)));
        }
    }

    private String resultsToString(List<LocalizedLabel> results) {
        sb.setLength(0); // set length of buffer to 0
        sb.trimToSize();
        int i = 0;
        for (LocalizedLabel result : results) {
            sb.append(result.getTitle()).append("\n");
            i++;
            if (i > 3) {
                break;
            }
        }
//        sb.append("move your finger around the screen to discover your environment.");
        return sb.toString();
    }
}

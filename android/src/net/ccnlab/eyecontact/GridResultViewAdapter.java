package net.ccnlab.eyecontact;

import android.content.Context;
import android.view.View;
import android.view.ViewGroup;
import android.view.accessibility.AccessibilityNodeInfo;
import android.widget.BaseAdapter;
import android.widget.TextView;

import net.ccnlab.eyecontact.env.Logger;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;


public class GridResultViewAdapter extends BaseAdapter {
    private List<Classifier.Recognition> results;
    private List<List<Classifier.Recognition>> resultsByGrid;
    private final StringBuilder sb;
    private List<TextView> textViews;
    private boolean clean = false;
    private static final int cellBackgroundColor = 0x99FFFFFF;

    GridResultViewAdapter(Context c) {
        super();
        this.results = new ArrayList<>();
        this.resultsByGrid = new ArrayList<>();
        textViews = new ArrayList<>();
        for (int i = 0; i < 16; i++) {
            resultsByGrid.add(i, new ArrayList<Classifier.Recognition>());
            TextView textView = new TextView(c);
            textViews.add(textView);
            // disables the "double click to activate" saying
            textView.setAccessibilityDelegate(new View.AccessibilityDelegate() {
                                                  @Override
                                                  public void onInitializeAccessibilityNodeInfo(View host, AccessibilityNodeInfo info) {
                                                      super.onInitializeAccessibilityNodeInfo(host, info);
                                                  }
                                              }
            );
            textView.setClickable(false);
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
        textView.setTextSize(15);
        textView.setBackgroundColor(cellBackgroundColor);
        textView.setHeight(viewGroup.getHeight() / 4);
        textView.setWidth(viewGroup.getWidth() / 4);
        return textView;
    }

    public List<Classifier.Recognition> getResults() {
        return results;
    }

    public void setResults(List<Classifier.Recognition> results) {
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

            while (last_index < results.size() && results.get(last_index).getGrid() == i) {
                this.resultsByGrid.get(i).add(results.get(last_index));
                last_index++;
            }

            Collections.sort(this.resultsByGrid.get(i), new Comparator<Classifier.Recognition>() {
                @Override
                public int compare(Classifier.Recognition lrec, Classifier.Recognition rrec) {
                    return Float.compare(rrec.getConfidence(), lrec.getConfidence());
                }
            });
            textViews.get(i).setText(resultsToString(this.resultsByGrid.get(i)));
        }
    }

    private String resultsToString(List<Classifier.Recognition> results) {
        sb.setLength(0); // set length of buffer to 0
        sb.trimToSize();

        for (Classifier.Recognition result : results) {
            sb.append(result.getTitle()).append("\n");
        }
        return sb.toString();
    }
}

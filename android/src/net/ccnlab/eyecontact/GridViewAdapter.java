package net.ccnlab.eyecontact;

import android.content.Context;
import android.view.View;
import android.view.ViewGroup;
import android.widget.BaseAdapter;
import android.widget.TextView;

import net.ccnlab.eyecontact.env.Logger;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Locale;


public class GridViewAdapter extends BaseAdapter {
    private List<Classifier.Recognition> results;
    private List<List<Classifier.Recognition>> resultsByGrid;
    private final StringBuilder sb;
    //    private static final Logger LOGGER = new Logger();
    private List<TextView> textViews;
    private List<Boolean> textViewAdded;
    private boolean clean = false;

    GridViewAdapter(Context c) {
        super();
        this.results = new ArrayList<>();
        this.resultsByGrid = new ArrayList<>();
        textViews = new ArrayList<>();
        textViewAdded = new ArrayList<>();
        for (int i = 0; i < 16; i++) {
            resultsByGrid.add(i, new ArrayList<Classifier.Recognition>());
            textViews.add(new TextView(c));
            textViewAdded.add(Boolean.FALSE);
        }
        this.sb = new StringBuilder();
//        super.getParent().findViewById(R.id.texture);
//        int width = c.get;
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
        if (textViewAdded.get(i))
            return textView;
        else {
            int top = 3;
            int bottom = 3;
            int left = 3;
            int right = 3;

            if (i / 4 == 0) {
                top = 0;
            } else if (i / 4 == 3) {
                bottom = 0;
            }
            if (i % 4 == 0) {
                left = 0;
            } else if (i % 4 == 3) {
                right = 0;
            }
            textView.setPadding(left, top, right, bottom);
            textView.setTextSize(15);
            textView.setBackgroundColor(0x330000FF);
            textView.setHeight((viewGroup.getHeight() - 18) / 4);
            textView.setWidth((viewGroup.getWidth() - 18) / 4);

            textViewAdded.set(i, Boolean.TRUE);
            return textView;
        }
    }

    public List<Classifier.Recognition> getResults() {
        return results;
    }

    public void setResults(List<Classifier.Recognition> results) {
        this.results = results;
    }

    void updateResults() {
//        ready = true;
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
        if (results.isEmpty()) {
            sb.append("empty");
        }

        for (Classifier.Recognition result : results) {
            sb.append(result.getTitle()).append(": ").
                    append(String.format(Locale.US, "%.2f", result.getConfidence())).append("\n");
        }
        return sb.toString();
    }
    public void setHeight(int height) {

    }
    public void setWidth(int width) {}
}

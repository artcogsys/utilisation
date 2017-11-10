package net.ccnlab.eyecontact;

import android.content.Context;
import android.widget.ArrayAdapter;

import net.ccnlab.eyecontact.model.Entity;

import java.util.List;

public class EntityArrayAdapter extends ArrayAdapter<Entity> {

    public EntityArrayAdapter(Context context, int resource, List<Entity> objects) {
        super(context, resource, objects);
    }
}

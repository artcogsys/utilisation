package net.ccnlab.eyecontact;

import android.app.ListActivity;
import android.os.Bundle;

import net.ccnlab.eyecontact.model.Entity;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;

public class ClassSelectionActivity extends ListActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_class_selection);
        Entity rootEntity = null;
        try {
            rootEntity = this.initializeClasses();
        } catch (IOException e) {
            e.printStackTrace();
        }
        EntityArrayAdapter entityArrayAdapter = new EntityArrayAdapter(this, R.id.class_selection_list_view, rootEntity.getSubEntities());
        setListAdapter(entityArrayAdapter);
    }

    private Entity initializeClasses() throws IOException{
        BufferedReader br = new BufferedReader(new InputStreamReader(getAssets().open("class_tree.txt")));
        String line;

        Entity root = new Entity("root");
        Entity current = root;
        while ((line = br.readLine()) != null) {
            current = parseEntity(line, current);
        }
        return root;
    }

    private Entity parseEntity(String entityLine, Entity previous) {
        String entityText = entityLine.trim();
        int entityLevel = entityLine.indexOf(entityText);
        Entity parent = previous;
        while (parent.getLevel() != entityLevel) {
            parent = parent.getParent();
        }
        int commaPos = entityText.indexOf(':');
        String name;
        int classId = -1;
        if (commaPos != -1) {
            name = entityText.substring(0, commaPos);
            classId = Integer.parseInt(entityText.substring(commaPos));
        } else {
            name = entityText;
        }

        Entity entity = new Entity(name, entityLevel, classId);
        parent.addChild(entity);

        return entity;
    }
}

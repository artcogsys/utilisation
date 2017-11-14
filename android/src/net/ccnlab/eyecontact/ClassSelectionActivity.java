package net.ccnlab.eyecontact;

import android.app.ListActivity;
import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.widget.AdapterView;
import android.widget.ListView;

import net.ccnlab.eyecontact.model.Entity;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;

// @TODO: Navigation is hacky, but it works. There needs to be a better way to do it.
public class ClassSelectionActivity extends ListActivity {

    Entity currentEntity;
    Entity rootEntity;
    EntityArrayAdapter entityArrayAdapter;
    ListView listView;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_class_selection);
        rootEntity = null;
        try {
            rootEntity = this.initializeClasses();
        } catch (IOException e) {
            e.printStackTrace();
        }
        entityArrayAdapter = new EntityArrayAdapter(this,
                R.layout.class_list_entity_text_view,
                android.R.id.text1,
                new ArrayList<Entity>());

        listView = (ListView) findViewById(android.R.id.list);

        listView.setOnItemClickListener(new AdapterView.OnItemClickListener() {
            @Override
            public void onItemClick(AdapterView<?> parent, View view, int position, long id) {
                Entity clickedEntity = currentEntity.getSubEntities().get(position);
                if(clickedEntity.hasClassId()) {
                    startClassifierActivity(clickedEntity.getClassId());
                } else {
                    displayEntity(clickedEntity);
                }
            }
        });

        setListAdapter(entityArrayAdapter);
        displayEntity(rootEntity);
    }

    private void startClassifierActivity(int classId){
        Intent intent = new Intent(this, ClassifierActivity.class);
        intent.putExtra("classId", classId);
        startActivity(intent);
    }


    @Override
    public void onBackPressed() {
        if (currentEntity.equals(rootEntity)) {
            super.onBackPressed();
        } else {
            displayEntity(currentEntity.getParent());
        }
    }

    private void displayEntity(Entity entity) {
        entityArrayAdapter.clear();
        entityArrayAdapter.addAll(entity.getSubEntities());
        currentEntity = entity;
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
        Entity entity;

        if (entityText.equals("")) {
            entity = previous;
        } else {
            int entityLevel = entityLine.indexOf(entityText);
            Entity parent = previous;
            while (parent.getLevel() + 4 != entityLevel && parent.getLevel() != -4) {
                parent = parent.getParent();
            }
            int commaPos = entityText.indexOf(':');
            String name;
            int classId = -1;
            if (commaPos != -1) {
                name = entityText.substring(0, commaPos);
                classId = Integer.parseInt(entityText.substring(commaPos + 1));
            } else {
                name = entityText;
            }
            entity = new Entity(name, entityLevel, classId);
            parent.addChild(entity);
            entity.setParent(parent);
        }
        return entity;
    }
}

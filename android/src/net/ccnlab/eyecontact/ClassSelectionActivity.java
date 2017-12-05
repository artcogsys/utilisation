package net.ccnlab.eyecontact;

import android.app.ListActivity;
import android.app.VoiceInteractor;
import android.app.VoiceInteractor.PickOptionRequest;
import android.app.VoiceInteractor.PickOptionRequest.Option;
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
                doRouting(clickedEntity);
            }
        });

        setListAdapter(entityArrayAdapter);
        doRouting(rootEntity);

    }

    private void doRouting(Entity clickedEntity) {
        if (clickedEntity.hasClassId()) {
            startClassifierActivity(clickedEntity);
        } else {
            displayEntity(clickedEntity);
        }
    }

    private void startClassifierActivity(Entity entityToFind) {
        Intent intent = new Intent(this, ClassifierActivity.class);
        intent.putExtra("classId", entityToFind.getClassId());
        intent.putExtra("classLabel", entityToFind.getFullName());
        startActivity(intent);
    }

    @Override
    public void onBackPressed() {
        if (currentEntity.equals(rootEntity)) {
            super.onBackPressed();
        } else {
            doRouting(currentEntity.getParent());
        }
    }

    private void displayEntity(Entity entity) {
        entityArrayAdapter.clear();
        entityArrayAdapter.addAll(entity.getSubEntities());
        currentEntity = entity;
        if (isVoiceInteraction()) {
            for (VoiceInteractor.Request voiceInteractorRequest : getVoiceInteractor().getActiveRequests()) {
                voiceInteractorRequest.cancel();
            }
            createVoiceInteractorRequest(currentEntity);
        }
    }

    private void createVoiceInteractorRequest(Entity entity) {
        int index = 0;
        Option[] options = new Option[entity.getSubEntities().size()];
        for (Entity subEntity : entity.getSubEntities()) {
            Option option = new Option(subEntity.getFirstName(), index);
            for (String synonym : subEntity.getSynonyms()) {
                option.addSynonym(synonym);
            }
            options[index] = option;
            index += 1;
        }
        getVoiceInteractor().submitRequest(new PickOptionRequest(new VoiceInteractor.Prompt("Choose a class"), options, null) {
            @Override
            public void onPickOptionResult(boolean finished, Option[] selections, Bundle result) {
                if (finished && selections.length == 1) {
                    doRouting(currentEntity.getSubEntities().get(selections[0].getIndex()));
                }
            }
        });
    }

    private Entity initializeClasses() throws IOException {
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

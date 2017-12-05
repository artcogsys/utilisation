package net.ccnlab.eyecontact.model;

import java.util.ArrayList;
import java.util.List;

public class Entity {
    private Entity parent;
    private List<Entity> subEntities;
    private String fullName;
    private int classId;
    private int level;
    private String firstName;
    private String[] synonyms;

    public Entity() {
        this.subEntities = new ArrayList<>();
    }

    public Entity(String fullName) {
        this(fullName, -4, -1);
    }

    public Entity(String fullName, int level, int classId) {
        this();
        setFullName(fullName);
        this.level = level;
        this.classId = classId;
    }

    public Entity getParent() {
        return parent;
    }

    public void setParent(Entity parent) {
        this.parent = parent;
    }

    public List<Entity> getSubEntities() {
        return subEntities;
    }

    public String getFullName() {
        return fullName;
    }

    public void setFullName(String fullName) {
        this.fullName = fullName;
        String[] tempAllNames = fullName.split(",");
        this.synonyms = new String[tempAllNames.length - 1];
        boolean first = true;
        int synonymIndex = 0;
        for (String name : tempAllNames) {
            if (first) {
                first = false;
                this.firstName = name.trim();
            } else {
                this.synonyms[synonymIndex] = name.trim();
                synonymIndex += 1;
            }
        }
    }

    public String getFirstName() {
        return firstName;
    }

    public String[] getSynonyms() {
        return synonyms;
    }

    public int getClassId() {
        return classId;
    }

    public void setClassId(int classId) {
        this.classId = classId;
    }

    public int getLevel() {
        return level;
    }

    public void setLevel(int level) {
        this.level = level;
    }

    public void addChild(Entity child) {
        subEntities.add(child);
    }

    public boolean hasClassId() {
        return this.classId != -1;
    }
    @Override
    public String toString() {
        return fullName;
    }
}

package net.ccnlab.eyecontact.model;

import java.util.ArrayList;
import java.util.List;

public class Entity {
    private Entity parent;
    private List<Entity> subEntities;
    private String name;
    private int classId;
    private int level;

    public Entity() {
        this.subEntities = new ArrayList<>();
    }

    public Entity(String name) {
        this(name, -4, -1);
    }

    public Entity(String name, int level, int classId) {
        this();
        this.name = name;
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

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
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
        return name;
    }
}

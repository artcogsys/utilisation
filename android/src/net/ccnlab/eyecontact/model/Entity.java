package net.ccnlab.eyecontact.model;

import java.util.ArrayList;
import java.util.LinkedList;
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

    // Clone Constructor for Category. Really ugly piece of code here.
    public Entity(Entity entity) {
        this();
        this.parent = entity.getParent();
        this.fullName = "Search everything belonging to, " + entity.getFullName();
        this.subEntities.clear();
        this.subEntities.addAll(entity.getSubEntities());
        this.classId = entity.getClassId();
        this.level = entity.getLevel();
        this.firstName = entity.getFirstName();
        this.synonyms = entity.getSynonyms();
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

    public void setSubEntities(List<Entity> subEntities) {
        this.subEntities.clear();
        this.subEntities.addAll((subEntities));
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

    public int[] getClassIds() {
        LinkedList<Integer> subClassIds = new LinkedList<>();
        aggregateSubClassIds(this, subClassIds);
        return toIntArray(subClassIds);
    }

    private int getClassId() {
        return classId;
    }

    public void setClassId(int classId) {
        this.classId = classId;
    }

    private void aggregateSubClassIds(Entity entity, List<Integer> classIds) {
        if (entity.getSubEntities().isEmpty()) {
            classIds.add(entity.getClassId());
        } else {
            for (Entity subEntity : entity.getSubEntities()) {
                aggregateSubClassIds(subEntity, classIds);
            }
        }
    }

    private int[] toIntArray(List<Integer> list) {
        int[] ret = new int[list.size()];
        int i = 0;
        for (Integer e : list)
            ret[i++] = e;
        return ret;
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

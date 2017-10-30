package net.ccnlab.eyecontact.model;

public class LocalizedLabel {

    private final String id;
    private final String title;
    private final Float confidence;
    private final Integer gridLocation;


    public LocalizedLabel(
            final String id, final String title, final Float confidence, final Integer gridLocation) {
        this.id = id;
        this.title = title;
        this.confidence = confidence;
        this.gridLocation = gridLocation;
    }

    public String getId() {
        return id;
    }

    public String getTitle() {
        return title;
    }

    public Float getConfidence() {
        return confidence;
    }

    public Integer getGridLocation() {
        return gridLocation;
    }

    @Override
    public String toString() {
        String resultString = "";
        if (id != null) {
            resultString += "[" + id + "] ";
        }

        if (title != null) {
            resultString += title + " ";
        }

        if (confidence != null) {
            resultString += String.format("(%.1f%%) ", confidence * 100.0f);
        }

        return resultString.trim();
    }
}

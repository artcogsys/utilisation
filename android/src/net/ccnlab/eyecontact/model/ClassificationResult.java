package net.ccnlab.eyecontact.model;

public class ClassificationResult {

    private final String title;
    private final Float confidence;

    public ClassificationResult(final String title, final Float confidence) {
        this.title = title;
        this.confidence = confidence;
    }

    public String getTitle() {
        return title;
    }

    public Float getConfidence() {
        return confidence;
    }

    @Override
    public String toString() {
        String resultString = "";

        if (title != null) {
            resultString += title + " ";
        }

        if (confidence != null) {
            resultString += String.format("(%.1f%%) ", confidence * 100.0f);
        }
        return resultString.trim();
    }

}

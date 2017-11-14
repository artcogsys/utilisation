package net.ccnlab.eyecontact.model;

import net.ccnlab.eyecontact.env.Logger;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.PriorityQueue;
import java.util.Vector;

public class ResultsContainer {
    public static final float LOCALIZED_LABEL_THRESHOLD = 0.5f;
    public static final float CLASSIFICATION_THRESHOLD = 0.40f;
    public static final int MAX_CLASSIFICATION_RESULTS = 4;
    private static final Logger LOGGER = new Logger();


    List<ClassificationResult> classifications;
    List<LocalizedLabel> localizedLabels;
    Vector<String> classificationLabelNames;
    Vector<String> localizedLabelNames;

    private Integer classIdToFind;
    private String labelOfClassToFind;

    public ResultsContainer(Vector<String> classificationLabelNames, Vector<String> localizedLabelNames,
                            Integer classIdToFind, String labelOfClassToFind) {
        this.classificationLabelNames = classificationLabelNames;
        this.localizedLabelNames = localizedLabelNames;
        this.classifications = new ArrayList<>();
        this.localizedLabels = new ArrayList<>();
        this.classIdToFind = classIdToFind;
        this.labelOfClassToFind = labelOfClassToFind;
    }

    public void setLocalizedLabelResults(float[] outputs) {
        for (int i = 0; i < outputs.length; ++i) {
            int object_id = i % 150;
            int object_location = i / 150;
            if (outputs[i] > LOCALIZED_LABEL_THRESHOLD) {
                localizedLabels.add(
                        new LocalizedLabel(
                                "" + i,
                                localizedLabelNames.get(object_id),
                                outputs[i],
                                object_location));
            }
        }
    }

    public void setClassificationResults(float[] outputs) {
        classifications.add(new ClassificationResult(
                "" + this.classIdToFind,
                this.labelOfClassToFind,
                outputs[this.classIdToFind]));
    }

    public List<ClassificationResult> getClassifications() {
        return classifications;
    }

    public void setClassifications(List<ClassificationResult> classifications) {
        this.classifications = classifications;
    }

    public List<LocalizedLabel> getLocalizedLabels() {
        return localizedLabels;
    }

    public void setLocalizedLabels(List<LocalizedLabel> localizedLabels) {
        this.localizedLabels = localizedLabels;
    }
}

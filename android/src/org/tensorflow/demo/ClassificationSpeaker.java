package org.tensorflow.demo;

import android.content.Context;
import android.speech.tts.TextToSpeech;
import android.speech.tts.UtteranceProgressListener;

import org.tensorflow.demo.env.Logger;

import java.util.List;
import java.util.Locale;

public class ClassificationSpeaker {
  private static final Logger LOGGER = new Logger();
  TextToSpeech textToSpeech;

  public void initialize(Context applicationContext) {
    LOGGER.i("Initializing TestToSpeech");
    textToSpeech = new TextToSpeech(applicationContext, new TextToSpeech.OnInitListener() {
      @Override
      public void onInit(int status) {
        if (status != TextToSpeech.ERROR) {
          textToSpeech.setLanguage(Locale.UK);
          textToSpeech.setOnUtteranceProgressListener(new UtteranceProgressListener() {
            @Override
            public void onStart(String s) {
              return;
            }

            @Override
            public void onDone(String s) {
              return;
            }

            @Override
            public void onError(String s) {
              return;
            }
          });
        }
      }
    });
    LOGGER.i("Text To Speech initialized");
  }

  public void speak(List<Classifier.Recognition> results) {

    for (Classifier.Recognition result : results) {
      if (result.getConfidence() > 0.5) {
        if (!textToSpeech.isSpeaking()) {
          textToSpeech.speak(result.getTitle(), TextToSpeech.QUEUE_FLUSH, null, TextToSpeech.Engine.KEY_PARAM_UTTERANCE_ID);
        }
      }
    }
  }

}

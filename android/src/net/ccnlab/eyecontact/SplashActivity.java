package net.ccnlab.eyecontact;

import android.app.Activity;
import android.os.Bundle;
import android.content.Intent;
import android.speech.RecognizerIntent;

public class SplashActivity extends Activity {


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        Intent intent = new Intent(this, ClassSelectionActivity.class);
        intent.setAction(Intent.ACTION_MAIN);
//        intent.putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM);
        startActivity(intent);
        finish();
    }

    @Override
    public void finish() {
        super.finish();
    }
}
package net.ccnlab.eyecontact;

import android.app.Activity;
import android.os.Bundle;
import android.content.Intent;

public class SplashActivity extends Activity {

    public static SplashActivity self;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        Intent intent = new Intent(this, ClassSelectionActivity.class);
        startActivity(intent);
        finish();
    }

    @Override
    public void finish() {
        super.finish();
    }
}
package org.tensorflow.lite.examples.detection;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Context;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.RectF;
import android.os.Bundle;
import android.os.Handler;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import org.tensorflow.lite.examples.detection.customview.OverlayView;
import org.tensorflow.lite.examples.detection.env.ImageUtils;
import org.tensorflow.lite.examples.detection.env.Logger;
import org.tensorflow.lite.examples.detection.env.Utils;
import org.tensorflow.lite.examples.detection.tflite.Classifier;
import org.tensorflow.lite.examples.detection.tflite.YoloV4Classifier;
import org.tensorflow.lite.examples.detection.tracking.MultiBoxTracker;

import java.io.IOException;
import java.util.LinkedList;
import java.util.List;

public class MainActivity extends AppCompatActivity {

    public static final float MINIMUM_CONFIDENCE_TF_OD_API = 0.5f;
    public static String selectedCameraType = "Rear Camera";  // Default
    public static String selectedModelFile = ModelCatalog.DEFAULT_MODEL_FILE;

    private List<String> availableModelFiles;
    private TextView selectedCameraValue;
    private TextView selectedModelValue;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        cameraButton = findViewById(R.id.cameraButton);
        detectButton = findViewById(R.id.detectButton);
        imageView = findViewById(R.id.imageView);
        selectedCameraValue = findViewById(R.id.selectedCameraValue);
        selectedModelValue = findViewById(R.id.selectedModelValue);
        Button chooseCameraBtn = findViewById(R.id.chooseCamera);
        Button selectModelBtn = findViewById(R.id.selectModelButton);

        availableModelFiles = ModelCatalog.getAvailableModelFiles(getAssets());
        selectedModelFile = ModelCatalog.getSafeModelFile(getAssets(), selectedModelFile);
        bindSelectionSummary();

        chooseCameraBtn.setOnClickListener(v -> showCameraSelectionDialog());

        selectModelBtn.setOnClickListener(v -> showModelSelectionDialog());

        cameraButton.setOnClickListener(v -> {
            Intent intent = new Intent(MainActivity.this, DetectorActivity.class);
            intent.putExtra("CAMERA_TYPE", selectedCameraType);
            intent.putExtra("MODEL_FILE", selectedModelFile);
            startActivity(intent);
        });

        detectButton.setOnClickListener(v -> {
            Handler handler = new Handler();

            new Thread(() -> {
                final List<Classifier.Recognition> results = detector.recognizeImage(cropBitmap);
                handler.post(new Runnable() {
                    @Override
                    public void run() {
                        handleResult(cropBitmap, results);
                    }
                });
            }).start();

        });
        this.sourceBitmap = Utils.getBitmapFromAsset(MainActivity.this, "kite.jpg");

        this.cropBitmap = Utils.processBitmap(sourceBitmap, TF_OD_API_INPUT_SIZE);

        this.imageView.setImageBitmap(cropBitmap);

        initBox();
    }
    private void showCameraSelectionDialog() {
        String[] cameraOptions = {"Rear Camera", "Front Camera", "USB Camera"};

        new android.app.AlertDialog.Builder(this)
                .setTitle("Select Camera")
                .setItems(cameraOptions, (dialog, which) -> {
                    selectedCameraType = cameraOptions[which];
                    Toast.makeText(this, selectedCameraType + " selected", Toast.LENGTH_SHORT).show();
                    bindSelectionSummary();
                })
                .show();
    }

    private void showModelSelectionDialog() {
        final String[] modelOptions = new String[availableModelFiles.size()];
        for (int i = 0; i < availableModelFiles.size(); i++) {
            modelOptions[i] = ModelCatalog.toDisplayName(availableModelFiles.get(i));
        }

        new android.app.AlertDialog.Builder(this)
                .setTitle("Select Model")
                .setItems(modelOptions, (dialog, which) -> {
                    selectedModelFile = availableModelFiles.get(which);
                    Toast.makeText(this, modelOptions[which] + " selected", Toast.LENGTH_SHORT).show();
                    bindSelectionSummary();
                })
                .show();
    }

    private void bindSelectionSummary() {
        if (selectedCameraValue != null) {
            selectedCameraValue.setText(selectedCameraType);
        }
        if (selectedModelValue != null) {
            selectedModelValue.setText(ModelCatalog.toDisplayName(selectedModelFile));
        }
    }

    private static final Logger LOGGER = new Logger();

    public static final int TF_OD_API_INPUT_SIZE = 416;

    private static final boolean TF_OD_API_IS_QUANTIZED = false;

    private static final String TF_OD_API_MODEL_FILE = "yolov4-416-fp32.tflite";

    private static final String TF_OD_API_LABELS_FILE = "file:///android_asset/coco.txt";

    // Minimum detection confidence to track a detection.
    private static final boolean MAINTAIN_ASPECT = false;
    private Integer sensorOrientation = 90;

    private Classifier detector;

    private Matrix frameToCropTransform;
    private Matrix cropToFrameTransform;
    private MultiBoxTracker tracker;
    private OverlayView trackingOverlay;

    protected int previewWidth = 0;
    protected int previewHeight = 0;

    private Bitmap sourceBitmap;
    private Bitmap cropBitmap;

    private Button cameraButton, detectButton;
    private ImageView imageView;

    private void initBox() {
        selectedModelFile = ModelCatalog.getSafeModelFile(getAssets(), selectedModelFile);
        previewHeight = TF_OD_API_INPUT_SIZE;
        previewWidth = TF_OD_API_INPUT_SIZE;
        frameToCropTransform =
                ImageUtils.getTransformationMatrix(
                        previewWidth, previewHeight,
                        TF_OD_API_INPUT_SIZE, TF_OD_API_INPUT_SIZE,
                        sensorOrientation, MAINTAIN_ASPECT);

        cropToFrameTransform = new Matrix();
        frameToCropTransform.invert(cropToFrameTransform);

        tracker = new MultiBoxTracker(this);
        trackingOverlay = findViewById(R.id.tracking_overlay);
        trackingOverlay.addCallback(
                canvas -> tracker.draw(canvas));

        tracker.setFrameConfiguration(TF_OD_API_INPUT_SIZE, TF_OD_API_INPUT_SIZE, sensorOrientation);

        try {
            detector =
                    YoloV4Classifier.create(
                            getAssets(),
                            selectedModelFile,
                            TF_OD_API_LABELS_FILE,
                            TF_OD_API_IS_QUANTIZED);
        } catch (final IOException e) {
            e.printStackTrace();
            LOGGER.e(e, "Exception initializing classifier!");
            Toast toast =
                    Toast.makeText(
                            getApplicationContext(), "Classifier could not be initialized", Toast.LENGTH_SHORT);
            toast.show();
            finish();
        }
    }

    private void handleResult(Bitmap bitmap, List<Classifier.Recognition> results) {
        final Canvas canvas = new Canvas(bitmap);
        final Paint paint = new Paint();
        paint.setColor(Color.RED);
        paint.setStyle(Paint.Style.STROKE);
        paint.setStrokeWidth(2.0f);

        final List<Classifier.Recognition> mappedRecognitions =
                new LinkedList<Classifier.Recognition>();

        for (final Classifier.Recognition result : results) {
            final RectF location = result.getLocation();
            if (location != null && result.getConfidence() >= MINIMUM_CONFIDENCE_TF_OD_API) {
                canvas.drawRect(location, paint);
//                cropToFrameTransform.mapRect(location);
//
//                result.setLocation(location);
//                mappedRecognitions.add(result);
            }
        }
//        tracker.trackResults(mappedRecognitions, new Random().nextInt());
//        trackingOverlay.postInvalidate();
        imageView.setImageBitmap(bitmap);
    }
}

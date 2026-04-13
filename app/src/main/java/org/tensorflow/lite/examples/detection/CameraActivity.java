/*
 * Copyright 2019 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tensorflow.lite.examples.detection;

import android.Manifest;
import android.annotation.SuppressLint;
import android.app.Fragment;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.hardware.Camera;
import android.hardware.camera2.CameraAccessException;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraManager;
import android.hardware.camera2.params.StreamConfigurationMap;
import android.location.Location;
import android.location.LocationManager;
import android.media.Image;
import android.media.Image.Plane;
import android.media.ImageReader;
import android.media.ImageReader.OnImageAvailableListener;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.Trace;
import android.util.Size;
import android.view.MotionEvent;
import android.view.Surface;
import android.view.View;
import android.view.ViewGroup;
import android.view.WindowManager;
import android.webkit.JavascriptInterface;
import android.webkit.WebChromeClient;
import android.webkit.WebSettings;
import android.webkit.WebView;
import android.webkit.WebViewClient;
import android.widget.ArrayAdapter;
import android.widget.AutoCompleteTextView;
import android.widget.CompoundButton;
import android.widget.FrameLayout;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.widget.SwitchCompat;
import androidx.core.content.ContextCompat;
import androidx.core.view.WindowCompat;
import androidx.core.view.WindowInsetsCompat;
import androidx.core.view.WindowInsetsControllerCompat;

import com.google.android.material.button.MaterialButton;

import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

import org.tensorflow.lite.examples.detection.env.ImageUtils;
import org.tensorflow.lite.examples.detection.env.Logger;

public abstract class CameraActivity extends AppCompatActivity
    implements OnImageAvailableListener,
        Camera.PreviewCallback,
        CompoundButton.OnCheckedChangeListener,
        View.OnClickListener {
  private static final Logger LOGGER = new Logger();

  private static final int PERMISSIONS_REQUEST = 1;
  private static final String[] REQUIRED_PERMISSIONS = {
    Manifest.permission.CAMERA,
    Manifest.permission.ACCESS_FINE_LOCATION,
    Manifest.permission.ACCESS_COARSE_LOCATION
  };
  private static final float DRAG_TAP_THRESHOLD = 20f;
  private static final double DEFAULT_LATITUDE = 20.5937;
  private static final double DEFAULT_LONGITUDE = 78.9629;
  private static final String MAP_MODE_SATELLITE = "satellite";

  protected int previewWidth = 0;
  protected int previewHeight = 0;
  protected String selectedCameraType = "Rear Camera";
  protected String selectedModelFile = ModelCatalog.DEFAULT_MODEL_FILE;
  protected TextView modelValueTextView;
  protected TextView cameraValueTextView;

  private boolean debug = false;
  private Handler handler;
  private HandlerThread handlerThread;
  private boolean useCamera2API;
  private boolean isProcessingFrame = false;
  private final byte[][] yuvBytes = new byte[3][];
  private int[] rgbBytes = null;
  private int yRowStride;
  private Runnable postInferenceCallback;
  private Runnable imageConverter;

  private SwitchCompat apiSwitchCompat;
  private TextView threadsTextView;
  private ImageView plusImageView;
  private ImageView minusImageView;
  private AutoCompleteTextView modelDropdown;

  private FrameLayout mapOverlayCard;
  private View mapOverlayTouchLayer;
  private FrameLayout mapFullscreenContainer;
  private WebView mapOverlayWebView;
  private WebView mapFullscreenWebView;
  private TextView mapDistanceValue;
  private TextView mapFullscreenDistanceChip;

  private float mapTouchDownRawX;
  private float mapTouchDownRawY;
  private float mapTouchStartX;
  private float mapTouchStartY;
  private boolean isDraggingMapCard;

  private Double userLatitude = DEFAULT_LATITUDE;
  private Double userLongitude = DEFAULT_LONGITUDE;
  private Double pointALatitude;
  private Double pointALongitude;
  private Double pointBLatitude;
  private Double pointBLongitude;
  private String pendingPointMode = "";
  private String currentMapMode = MAP_MODE_SATELLITE;
  private String latestDistanceText = "Tap two points to measure";
  private boolean mapOverlayReady;
  private boolean mapFullscreenReady;

  @Override
  protected void onCreate(final Bundle savedInstanceState) {
    LOGGER.d("onCreate " + this);
    super.onCreate(null);

    selectedCameraType = getIntent().getStringExtra("CAMERA_TYPE");
    if (selectedCameraType == null) {
      selectedCameraType = "Rear Camera";
    }
    selectedModelFile =
        ModelCatalog.getSafeModelFile(getAssets(), getIntent().getStringExtra("MODEL_FILE"));

    getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
    WindowCompat.setDecorFitsSystemWindows(getWindow(), false);
    setContentView(R.layout.tfe_od_activity_camera);
    applyImmersiveMode();

    bindTopControls();
    bindSettingsRail();
    bindMapViews();

    if (hasAllPermissions()) {
      loadUserLocation();
      setFragment();
    } else {
      requestPermissions(REQUIRED_PERMISSIONS, PERMISSIONS_REQUEST);
    }

    showActiveModel(ModelCatalog.toDisplayName(selectedModelFile));
    showSelectedCamera(selectedCameraType);
    updateDistanceLabels(latestDistanceText);
  }

  private void bindTopControls() {
    final View backButton = findViewById(R.id.backButton);
    if (backButton != null) {
      backButton.setOnClickListener(v -> finish());
    }

    modelDropdown = findViewById(R.id.model_dropdown);
    if (modelDropdown == null) {
      return;
    }

    final List<String> modelFiles = ModelCatalog.getAvailableModelFiles(getAssets());
    final List<String> displayNames = new ArrayList<>();
    for (final String modelFile : modelFiles) {
      displayNames.add(ModelCatalog.toDisplayName(modelFile));
    }

    final ArrayAdapter<String> adapter =
        new ArrayAdapter<>(this, android.R.layout.simple_list_item_1, displayNames);
    modelDropdown.setAdapter(adapter);
    modelDropdown.setDropDownBackgroundDrawable(
        ContextCompat.getDrawable(this, R.drawable.bg_dropdown_menu));
    modelDropdown.setText(ModelCatalog.toDisplayName(selectedModelFile), false);
    modelDropdown.setOnClickListener(v -> modelDropdown.showDropDown());
    modelDropdown.setOnItemClickListener(
        (parent, view, position, id) -> {
          final String newModelFile = modelFiles.get(position);
          if (!newModelFile.equals(selectedModelFile)) {
            restartWithModel(newModelFile);
          }
        });
  }

  private void bindSettingsRail() {
    modelValueTextView = findViewById(R.id.model_value);
    cameraValueTextView = findViewById(R.id.camera_value);
    threadsTextView = findViewById(R.id.threads);
    plusImageView = findViewById(R.id.plus);
    minusImageView = findViewById(R.id.minus);
    apiSwitchCompat = findViewById(R.id.api_info_switch);
    if (apiSwitchCompat != null) {
      apiSwitchCompat.setOnCheckedChangeListener(this);
    }
    if (plusImageView != null) {
      plusImageView.setOnClickListener(this);
    }
    if (minusImageView != null) {
      minusImageView.setOnClickListener(this);
    }
  }

  private void bindMapViews() {
    mapOverlayCard = findViewById(R.id.map_overlay_card);
    mapOverlayTouchLayer = findViewById(R.id.map_overlay_touch_layer);
    mapFullscreenContainer = findViewById(R.id.map_fullscreen_container);
    mapOverlayWebView = findViewById(R.id.map_overlay_webview);
    mapFullscreenWebView = findViewById(R.id.map_fullscreen_webview);
    mapDistanceValue = findViewById(R.id.map_distance_value);
    mapFullscreenDistanceChip = findViewById(R.id.map_fullscreen_distance_chip);

    configureMapWebView(mapOverlayWebView, true);
    configureMapWebView(mapFullscreenWebView, false);

    if (mapOverlayTouchLayer != null) {
      mapOverlayTouchLayer.setOnTouchListener(this::handleMapOverlayTouch);
    }

    bindMapButton(R.id.map_close_button, v -> closeFullscreenMap());
    bindMapButton(R.id.map_mark_a_button, v -> setPendingPointMode("A"));
    bindMapButton(R.id.map_mark_b_button, v -> setPendingPointMode("B"));
    bindMapButton(
        R.id.map_clear_button,
        v -> {
          pointALatitude = null;
          pointALongitude = null;
          pointBLatitude = null;
          pointBLongitude = null;
          setPendingPointMode("");
          syncMapState();
        });
    bindMapButton(R.id.map_my_location_button, v -> focusOnUserLocation());
    bindMapButton(
        R.id.map_satellite_button,
        v -> {
          currentMapMode = MAP_MODE_SATELLITE;
          syncMapState();
        });
  }

  private void bindMapButton(final int viewId, final View.OnClickListener listener) {
    final View view = findViewById(viewId);
    if (view != null) {
      view.setOnClickListener(listener);
    }
  }

  private boolean handleMapOverlayTouch(final View view, final MotionEvent event) {
    if (mapOverlayCard == null || mapFullscreenContainer == null) {
      return false;
    }

    switch (event.getActionMasked()) {
      case MotionEvent.ACTION_DOWN:
        mapTouchDownRawX = event.getRawX();
        mapTouchDownRawY = event.getRawY();
        mapTouchStartX = mapOverlayCard.getX();
        mapTouchStartY = mapOverlayCard.getY();
        isDraggingMapCard = false;
        return true;
      case MotionEvent.ACTION_MOVE:
        final float deltaX = event.getRawX() - mapTouchDownRawX;
        final float deltaY = event.getRawY() - mapTouchDownRawY;
        if (Math.abs(deltaX) > DRAG_TAP_THRESHOLD || Math.abs(deltaY) > DRAG_TAP_THRESHOLD) {
          isDraggingMapCard = true;
        }
        positionMapOverlay(mapTouchStartX + deltaX, mapTouchStartY + deltaY);
        return true;
      case MotionEvent.ACTION_UP:
        if (!isDraggingMapCard) {
          openFullscreenMap();
        }
        return true;
      default:
        return false;
    }
  }

  private void positionMapOverlay(final float targetX, final float targetY) {
    if (mapOverlayCard == null || !(mapOverlayCard.getParent() instanceof ViewGroup)) {
      return;
    }

    final ViewGroup parent = (ViewGroup) mapOverlayCard.getParent();
    final float maxX = Math.max(0, parent.getWidth() - mapOverlayCard.getWidth());
    final float maxY = Math.max(0, parent.getHeight() - mapOverlayCard.getHeight());
    mapOverlayCard.setX(Math.max(0, Math.min(targetX, maxX)));
    mapOverlayCard.setY(Math.max(0, Math.min(targetY, maxY)));
  }

  private void openFullscreenMap() {
    if (mapFullscreenContainer != null) {
      mapFullscreenContainer.setVisibility(View.VISIBLE);
    }
    if (mapOverlayCard != null) {
      mapOverlayCard.setVisibility(View.GONE);
    }
    syncMapState();
  }

  private void closeFullscreenMap() {
    if (mapFullscreenContainer != null) {
      mapFullscreenContainer.setVisibility(View.GONE);
    }
    if (mapOverlayCard != null) {
      mapOverlayCard.setVisibility(View.VISIBLE);
    }
    syncMapState();
  }

  @SuppressLint("SetJavaScriptEnabled")
  private void configureMapWebView(final WebView webView, final boolean compactMode) {
    if (webView == null) {
      return;
    }

    final WebSettings settings = webView.getSettings();
    settings.setJavaScriptEnabled(true);
    settings.setDomStorageEnabled(true);
    settings.setLoadWithOverviewMode(true);
    settings.setUseWideViewPort(true);
    settings.setBuiltInZoomControls(false);
    settings.setDisplayZoomControls(false);

    webView.setBackgroundColor(0x00000000);
    webView.addJavascriptInterface(new MapJavascriptBridge(), "AndroidBridge");
    webView.setWebChromeClient(new WebChromeClient());
    webView.setWebViewClient(
        new WebViewClient() {
          @Override
          public void onPageFinished(final WebView view, final String url) {
            if (view == mapOverlayWebView) {
              mapOverlayReady = true;
            } else if (view == mapFullscreenWebView) {
              mapFullscreenReady = true;
            }
            runJavascript(
                view,
                String.format(Locale.US, "window.setCompactMode(%s);", compactMode ? "true" : "false"));
            syncMapState();
          }
        });
    webView.loadUrl("file:///android_asset/map_overlay.html");
  }

  private void setPendingPointMode(final String mode) {
    pendingPointMode = mode;
    syncMapState();
  }

  private void focusOnUserLocation() {
    loadUserLocation();
    syncMapState();
    runJavascript(mapFullscreenWebView, "window.focusOnUser();");
    runJavascript(mapOverlayWebView, "window.focusOnUser();");
  }

  private void syncMapState() {
    final String stateCommand =
        String.format(
            Locale.US,
            "window.applyState({userLat:%s,userLng:%s,pointALat:%s,pointALng:%s,pointBLat:%s,pointBLng:%s,pendingMode:'%s',mapMode:'%s'});",
            valueOrNull(userLatitude),
            valueOrNull(userLongitude),
            valueOrNull(pointALatitude),
            valueOrNull(pointALongitude),
            valueOrNull(pointBLatitude),
            valueOrNull(pointBLongitude),
            pendingPointMode,
            currentMapMode);

    if (mapOverlayReady) {
      runJavascript(mapOverlayWebView, stateCommand);
    }
    if (mapFullscreenReady) {
      runJavascript(mapFullscreenWebView, stateCommand);
    }
    updateDistanceLabels(latestDistanceText);
  }

  private String valueOrNull(final Double value) {
    return value == null ? "null" : String.format(Locale.US, "%.6f", value);
  }

  private void runJavascript(final WebView webView, final String command) {
    if (webView != null) {
      webView.post(() -> webView.evaluateJavascript(command, null));
    }
  }

  private void restartWithModel(final String newModelFile) {
    final Intent intent = new Intent(this, DetectorActivity.class);
    intent.putExtra("CAMERA_TYPE", selectedCameraType);
    intent.putExtra("MODEL_FILE", newModelFile);
    startActivity(intent);
    finish();
    overridePendingTransition(0, 0);
  }

  private void applyImmersiveMode() {
    final WindowInsetsControllerCompat controller =
        WindowCompat.getInsetsController(getWindow(), getWindow().getDecorView());
    if (controller != null) {
      controller.hide(
          WindowInsetsCompat.Type.statusBars() | WindowInsetsCompat.Type.navigationBars());
      controller.setSystemBarsBehavior(
          WindowInsetsControllerCompat.BEHAVIOR_SHOW_TRANSIENT_BARS_BY_SWIPE);
    }
  }

  @SuppressLint("MissingPermission")
  private void loadUserLocation() {
    if (!hasLocationPermission()) {
      return;
    }

    final LocationManager locationManager =
        (LocationManager) getSystemService(Context.LOCATION_SERVICE);
    if (locationManager == null) {
      return;
    }

    Location bestLocation = null;
    final List<String> providers = locationManager.getProviders(true);
    for (final String provider : providers) {
      final Location location = locationManager.getLastKnownLocation(provider);
      if (location != null
          && (bestLocation == null || location.getAccuracy() < bestLocation.getAccuracy())) {
        bestLocation = location;
      }
    }

    if (bestLocation != null) {
      userLatitude = bestLocation.getLatitude();
      userLongitude = bestLocation.getLongitude();
    }
  }

  private boolean hasLocationPermission() {
    return checkSelfPermission(Manifest.permission.ACCESS_FINE_LOCATION)
            == PackageManager.PERMISSION_GRANTED
        || checkSelfPermission(Manifest.permission.ACCESS_COARSE_LOCATION)
            == PackageManager.PERMISSION_GRANTED;
  }

  protected int[] getRgbBytes() {
    imageConverter.run();
    return rgbBytes;
  }

  protected int getLuminanceStride() {
    return yRowStride;
  }

  protected byte[] getLuminance() {
    return yuvBytes[0];
  }

  @Override
  public void onPreviewFrame(final byte[] bytes, final Camera camera) {
    if (isProcessingFrame) {
      LOGGER.w("Dropping frame!");
      return;
    }

    try {
      if (rgbBytes == null) {
        final Camera.Size previewSize = camera.getParameters().getPreviewSize();
        previewHeight = previewSize.height;
        previewWidth = previewSize.width;
        rgbBytes = new int[previewWidth * previewHeight];
        onPreviewSizeChosen(new Size(previewSize.width, previewSize.height), 90);
      }
    } catch (final Exception e) {
      LOGGER.e(e, "Exception!");
      return;
    }

    isProcessingFrame = true;
    yuvBytes[0] = bytes;
    yRowStride = previewWidth;

    imageConverter =
        () -> {
          try {
            ImageUtils.convertYUV420SPToARGB8888(bytes, previewWidth, previewHeight, rgbBytes);
          } catch (final Exception e) {
            LOGGER.e(e, "Preview conversion failed");
          }
        };

    postInferenceCallback =
        () -> {
          camera.addCallbackBuffer(bytes);
          isProcessingFrame = false;
        };
    processImage();
  }

  @Override
  public void onImageAvailable(final ImageReader reader) {
    if (previewWidth == 0 || previewHeight == 0) {
      return;
    }
    if (rgbBytes == null) {
      rgbBytes = new int[previewWidth * previewHeight];
    }
    try {
      final Image image = reader.acquireLatestImage();
      if (image == null) {
        return;
      }
      if (isProcessingFrame) {
        image.close();
        return;
      }
      isProcessingFrame = true;
      Trace.beginSection("imageAvailable");
      final Plane[] planes = image.getPlanes();
      fillBytes(planes, yuvBytes);
      yRowStride = planes[0].getRowStride();
      final int uvRowStride = planes[1].getRowStride();
      final int uvPixelStride = planes[1].getPixelStride();

      imageConverter =
          () -> {
            try {
              ImageUtils.convertYUV420ToARGB8888(
                  yuvBytes[0],
                  yuvBytes[1],
                  yuvBytes[2],
                  previewWidth,
                  previewHeight,
                  yRowStride,
                  uvRowStride,
                  uvPixelStride,
                  rgbBytes);
            } catch (final Exception e) {
              LOGGER.e(e, "Image conversion failed");
            }
          };

      postInferenceCallback =
          () -> {
            image.close();
            isProcessingFrame = false;
          };

      processImage();
    } catch (final Exception e) {
      LOGGER.e(e, "Exception!");
      Trace.endSection();
      return;
    }
    Trace.endSection();
  }

  @Override
  public void onBackPressed() {
    if (mapFullscreenContainer != null && mapFullscreenContainer.getVisibility() == View.VISIBLE) {
      closeFullscreenMap();
      return;
    }
    super.onBackPressed();
  }

  @Override
  public synchronized void onResume() {
    LOGGER.d("onResume " + this);
    super.onResume();
    applyImmersiveMode();

    handlerThread = new HandlerThread("inference");
    handlerThread.start();
    handler = new Handler(handlerThread.getLooper());
  }

  @Override
  public synchronized void onPause() {
    LOGGER.d("onPause " + this);
    if (handlerThread != null) {
      handlerThread.quitSafely();
      try {
        handlerThread.join();
        handlerThread = null;
        handler = null;
      } catch (final InterruptedException e) {
        LOGGER.e(e, "Exception!");
      }
    }
    super.onPause();
  }

  protected synchronized void runInBackground(final Runnable r) {
    if (handler != null) {
      handler.post(r);
    }
  }

  @Override
  public void onRequestPermissionsResult(
      final int requestCode, final String[] permissions, final int[] grantResults) {
    super.onRequestPermissionsResult(requestCode, permissions, grantResults);
    if (requestCode == PERMISSIONS_REQUEST) {
      if (allPermissionsGranted(grantResults)) {
        loadUserLocation();
        setFragment();
        syncMapState();
      } else {
        Toast.makeText(this, R.string.permissions_required, Toast.LENGTH_LONG).show();
        requestPermissions(REQUIRED_PERMISSIONS, PERMISSIONS_REQUEST);
      }
    }
  }

  private static boolean allPermissionsGranted(final int[] grantResults) {
    for (final int result : grantResults) {
      if (result != PackageManager.PERMISSION_GRANTED) {
        return false;
      }
    }
    return true;
  }

  private boolean hasAllPermissions() {
    if (Build.VERSION.SDK_INT < Build.VERSION_CODES.M) {
      return true;
    }
    for (final String permission : REQUIRED_PERMISSIONS) {
      if (checkSelfPermission(permission) != PackageManager.PERMISSION_GRANTED) {
        return false;
      }
    }
    return true;
  }

  private boolean isHardwareLevelSupported(
      final CameraCharacteristics characteristics, final int requiredLevel) {
    final int deviceLevel =
        characteristics.get(CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL);
    if (deviceLevel == CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL_LEGACY) {
      return requiredLevel == deviceLevel;
    }
    return requiredLevel <= deviceLevel;
  }

  private String chooseCamera() {
    final CameraManager manager = (CameraManager) getSystemService(Context.CAMERA_SERVICE);
    try {
      for (final String cameraId : manager.getCameraIdList()) {
        final CameraCharacteristics characteristics = manager.getCameraCharacteristics(cameraId);
        final Integer facing = characteristics.get(CameraCharacteristics.LENS_FACING);
        if (selectedCameraType.equals("Front Camera")
            && facing != null
            && facing == CameraCharacteristics.LENS_FACING_FRONT) {
          return validateCamera(characteristics, cameraId);
        }
        if (selectedCameraType.equals("Rear Camera")
            && facing != null
            && facing == CameraCharacteristics.LENS_FACING_BACK) {
          return validateCamera(characteristics, cameraId);
        }
        if (selectedCameraType.equals("USB Camera")
            && facing != null
            && facing == CameraCharacteristics.LENS_FACING_EXTERNAL) {
          return validateCamera(characteristics, cameraId);
        }
      }
    } catch (final CameraAccessException e) {
      LOGGER.e(e, "Not allowed to access camera");
    }
    return null;
  }

  private String validateCamera(
      final CameraCharacteristics characteristics, final String cameraId) {
    final StreamConfigurationMap map =
        characteristics.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP);
    if (map == null) {
      return null;
    }
    useCamera2API =
        (characteristics.get(CameraCharacteristics.LENS_FACING)
                == CameraCharacteristics.LENS_FACING_EXTERNAL)
            || isHardwareLevelSupported(
                characteristics, CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL_FULL);
    LOGGER.i("Camera API lv2?: %s", useCamera2API);
    return cameraId;
  }

  protected void setFragment() {
    final String cameraId = chooseCamera();
    final Fragment fragment;
    if (useCamera2API) {
      final CameraConnectionFragment camera2Fragment =
          CameraConnectionFragment.newInstance(
              (size, rotation) -> {
                previewHeight = size.getHeight();
                previewWidth = size.getWidth();
                CameraActivity.this.onPreviewSizeChosen(size, rotation);
              },
              this,
              getLayoutId(),
              getDesiredPreviewFrameSize());
      camera2Fragment.setCamera(cameraId);
      fragment = camera2Fragment;
    } else {
      fragment =
          new LegacyCameraConnectionFragment(this, getLayoutId(), getDesiredPreviewFrameSize());
    }
    getFragmentManager().beginTransaction().replace(R.id.container, fragment).commit();
  }

  protected void fillBytes(final Plane[] planes, final byte[][] yuvBytes) {
    for (int i = 0; i < planes.length; ++i) {
      final ByteBuffer buffer = planes[i].getBuffer();
      if (yuvBytes[i] == null) {
        LOGGER.d("Initializing buffer %d at size %d", i, buffer.capacity());
        yuvBytes[i] = new byte[buffer.capacity()];
      }
      buffer.get(yuvBytes[i]);
    }
  }

  public boolean isDebug() {
    return debug;
  }

  protected void readyForNextImage() {
    if (postInferenceCallback != null) {
      postInferenceCallback.run();
    }
  }

  protected int getScreenOrientation() {
    switch (getWindowManager().getDefaultDisplay().getRotation()) {
      case Surface.ROTATION_270:
        return 270;
      case Surface.ROTATION_180:
        return 180;
      case Surface.ROTATION_90:
        return 90;
      default:
        return 0;
    }
  }

  @Override
  public void onCheckedChanged(final CompoundButton buttonView, final boolean isChecked) {
    setUseNNAPI(isChecked);
    if (apiSwitchCompat != null) {
      apiSwitchCompat.setText(isChecked ? "NNAPI" : "TFLITE");
    }
  }

  @Override
  public void onClick(final View v) {
    if (threadsTextView == null) {
      return;
    }
    if (v.getId() == R.id.plus) {
      final int numThreads = Integer.parseInt(threadsTextView.getText().toString().trim());
      if (numThreads >= 9) {
        return;
      }
      threadsTextView.setText(String.valueOf(numThreads + 1));
      setNumThreads(numThreads + 1);
    } else if (v.getId() == R.id.minus) {
      final int numThreads = Integer.parseInt(threadsTextView.getText().toString().trim());
      if (numThreads <= 1) {
        return;
      }
      threadsTextView.setText(String.valueOf(numThreads - 1));
      setNumThreads(numThreads - 1);
    }
  }

  protected void showActiveModel(final String modelName) {
    if (modelValueTextView != null) {
      modelValueTextView.setText(modelName);
    }
  }

  protected void showSelectedCamera(final String cameraName) {
    if (cameraValueTextView != null) {
      cameraValueTextView.setText(cameraName);
    }
  }

  protected void showFrameInfo(final String frameInfo) {}

  protected void showCropInfo(final String cropInfo) {}

  protected void showInference(final String inferenceTime) {}

  private void updateDistanceLabels(final String text) {
    latestDistanceText = text;
    if (mapDistanceValue != null) {
      mapDistanceValue.setText(text);
    }
    if (mapFullscreenDistanceChip != null) {
      mapFullscreenDistanceChip.setText(text);
    }
  }

  private final class MapJavascriptBridge {
    @JavascriptInterface
    public void onDistanceChanged(final String value) {
      runOnUiThread(() -> updateDistanceLabels(value));
    }

    @JavascriptInterface
    public void onPointChanged(
        final String pointId, final double latitude, final double longitude) {
      runOnUiThread(
          () -> {
            if ("A".equals(pointId)) {
              pointALatitude = latitude;
              pointALongitude = longitude;
            } else if ("B".equals(pointId)) {
              pointBLatitude = latitude;
              pointBLongitude = longitude;
            }
            pendingPointMode = "";
            syncMapState();
          });
    }
  }

  protected abstract void processImage();

  protected abstract void onPreviewSizeChosen(final Size size, final int rotation);

  protected abstract int getLayoutId();

  protected abstract Size getDesiredPreviewFrameSize();

  protected abstract void setNumThreads(int numThreads);

  protected abstract void setUseNNAPI(boolean isChecked);
}

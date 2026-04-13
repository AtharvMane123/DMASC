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
import android.app.Fragment;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.hardware.Camera;
import android.hardware.camera2.CameraAccessException;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraManager;
import android.hardware.camera2.params.StreamConfigurationMap;
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
import android.view.Surface;
import android.view.View;
import android.view.WindowManager;
import android.widget.ArrayAdapter;
import android.widget.AutoCompleteTextView;
import android.widget.CompoundButton;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.widget.SwitchCompat;
import androidx.core.content.ContextCompat;
import androidx.core.view.WindowCompat;
import androidx.core.view.WindowInsetsCompat;
import androidx.core.view.WindowInsetsControllerCompat;

import com.google.android.material.bottomsheet.BottomSheetBehavior;

import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;

import org.tensorflow.lite.examples.detection.env.ImageUtils;
import org.tensorflow.lite.examples.detection.env.Logger;

public abstract class CameraActivity extends AppCompatActivity
    implements OnImageAvailableListener,
        Camera.PreviewCallback,
        CompoundButton.OnCheckedChangeListener,
        View.OnClickListener {
  private static final Logger LOGGER = new Logger();

  private static final int PERMISSIONS_REQUEST = 1;
  private static final String PERMISSION_CAMERA = Manifest.permission.CAMERA;

  protected int previewWidth = 0;
  protected int previewHeight = 0;
  protected String selectedCameraType = "Rear Camera";
  protected String selectedModelFile = ModelCatalog.DEFAULT_MODEL_FILE;
  protected TextView frameValueTextView;
  protected TextView cropValueTextView;
  protected TextView inferenceTimeTextView;
  protected TextView modelValueTextView;
  protected TextView cameraValueTextView;
  protected ImageView bottomSheetArrowImageView;

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

  private LinearLayout bottomSheetLayout;
  private BottomSheetBehavior<LinearLayout> sheetBehavior;
  private ImageView plusImageView;
  private ImageView minusImageView;
  private SwitchCompat apiSwitchCompat;
  private TextView threadsTextView;
  private AutoCompleteTextView modelDropdown;

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

    bindTopOverlay();

    if (hasPermission()) {
      setFragment();
    } else {
      requestPermission();
    }

    threadsTextView = findViewById(R.id.threads);
    plusImageView = findViewById(R.id.plus);
    minusImageView = findViewById(R.id.minus);
    apiSwitchCompat = findViewById(R.id.api_info_switch);
    bottomSheetLayout = findViewById(R.id.bottom_sheet_layout);
    bottomSheetArrowImageView = findViewById(R.id.bottom_sheet_arrow);
    if (bottomSheetLayout != null) {
      sheetBehavior = BottomSheetBehavior.from(bottomSheetLayout);
      sheetBehavior.setHideable(false);
      sheetBehavior.setState(BottomSheetBehavior.STATE_EXPANDED);
      sheetBehavior.setDraggable(false);
      sheetBehavior.setBottomSheetCallback(
          new BottomSheetBehavior.BottomSheetCallback() {
            @Override
            public void onStateChanged(@NonNull final View bottomSheet, final int newState) {
              if (bottomSheetArrowImageView == null) {
                return;
              }
              if (newState == BottomSheetBehavior.STATE_EXPANDED) {
                bottomSheetArrowImageView.setImageResource(R.drawable.icn_chevron_down);
              } else {
                bottomSheetArrowImageView.setImageResource(R.drawable.icn_chevron_up);
              }
            }

            @Override
            public void onSlide(@NonNull final View bottomSheet, final float slideOffset) {}
          });
    }

    frameValueTextView = findViewById(R.id.frame_info);
    cropValueTextView = findViewById(R.id.crop_info);
    inferenceTimeTextView = findViewById(R.id.inference_info);
    modelValueTextView = findViewById(R.id.model_value);
    cameraValueTextView = findViewById(R.id.camera_value);

    if (apiSwitchCompat != null) {
      apiSwitchCompat.setOnCheckedChangeListener(this);
    }
    if (plusImageView != null) {
      plusImageView.setOnClickListener(this);
    }
    if (minusImageView != null) {
      minusImageView.setOnClickListener(this);
    }

    showActiveModel(ModelCatalog.toDisplayName(selectedModelFile));
    showSelectedCamera(selectedCameraType);
  }

  private void bindTopOverlay() {
    final View backButton = findViewById(R.id.backButton);
    if (backButton != null) {
      backButton.setOnClickListener(v -> finish());
    }

    modelDropdown = findViewById(R.id.model_dropdown);
    if (modelDropdown == null) {
      return;
    }

    final List<String> availableModelFiles = ModelCatalog.getAvailableModelFiles(getAssets());
    final List<String> displayNames = new ArrayList<>();
    for (final String modelFile : availableModelFiles) {
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
          final String newModelFile = availableModelFiles.get(position);
          if (!newModelFile.equals(selectedModelFile)) {
            restartWithModel(newModelFile);
          }
        });
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

    handlerThread.quitSafely();
    try {
      handlerThread.join();
      handlerThread = null;
      handler = null;
    } catch (final InterruptedException e) {
      LOGGER.e(e, "Exception!");
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
        setFragment();
      } else {
        requestPermission();
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

  private boolean hasPermission() {
    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
      return checkSelfPermission(PERMISSION_CAMERA) == PackageManager.PERMISSION_GRANTED;
    } else {
      return true;
    }
  }

  private void requestPermission() {
    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
      if (shouldShowRequestPermissionRationale(PERMISSION_CAMERA)) {
        Toast.makeText(
                CameraActivity.this,
                "Camera permission is required for this demo",
                Toast.LENGTH_LONG)
            .show();
      }
      requestPermissions(new String[] {PERMISSION_CAMERA}, PERMISSIONS_REQUEST);
    }
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
      fragment = new LegacyCameraConnectionFragment(this, getLayoutId(), getDesiredPreviewFrameSize());
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

  protected void showFrameInfo(final String frameInfo) {
    if (frameValueTextView != null) {
      frameValueTextView.setText(frameInfo);
    }
  }

  protected void showCropInfo(final String cropInfo) {
    if (cropValueTextView != null) {
      cropValueTextView.setText(cropInfo);
    }
  }

  protected void showInference(final String inferenceTime) {
    if (inferenceTimeTextView != null) {
      inferenceTimeTextView.setText(inferenceTime);
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

  protected abstract void processImage();

  protected abstract void onPreviewSizeChosen(final Size size, final int rotation);

  protected abstract int getLayoutId();

  protected abstract Size getDesiredPreviewFrameSize();

  protected abstract void setNumThreads(int numThreads);

  protected abstract void setUseNNAPI(boolean isChecked);
}

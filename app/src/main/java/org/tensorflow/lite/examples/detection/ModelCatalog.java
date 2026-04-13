package org.tensorflow.lite.examples.detection;

import android.content.res.AssetManager;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Locale;

public final class ModelCatalog {
    public static final String DEFAULT_MODEL_FILE = "yolov4-416-fp32.tflite";

    private ModelCatalog() {
    }

    public static List<String> getAvailableModelFiles(final AssetManager assetManager) {
        final List<String> modelFiles = new ArrayList<>();
        try {
            final String[] assetFiles = assetManager.list("");
            if (assetFiles != null) {
                for (final String assetFile : assetFiles) {
                    if (assetFile != null && assetFile.toLowerCase(Locale.US).endsWith(".tflite")) {
                        modelFiles.add(assetFile);
                    }
                }
            }
        } catch (final IOException ignored) {
        }

        if (!modelFiles.contains(DEFAULT_MODEL_FILE)) {
            modelFiles.add(DEFAULT_MODEL_FILE);
        }

        Collections.sort(modelFiles);
        return modelFiles;
    }

    public static String getSafeModelFile(final AssetManager assetManager, final String requestedModelFile) {
        final List<String> availableModelFiles = getAvailableModelFiles(assetManager);
        if (requestedModelFile != null && availableModelFiles.contains(requestedModelFile)) {
            return requestedModelFile;
        }
        return availableModelFiles.isEmpty() ? DEFAULT_MODEL_FILE : availableModelFiles.get(0);
    }

    public static String toDisplayName(final String modelFile) {
        if (modelFile == null || modelFile.trim().isEmpty()) {
            return "Default Model";
        }

        final String withoutExtension = modelFile.endsWith(".tflite")
                ? modelFile.substring(0, modelFile.length() - ".tflite".length())
                : modelFile;
        final String normalized = withoutExtension.replace('-', ' ').replace('_', ' ').trim();
        if (normalized.isEmpty()) {
            return modelFile;
        }

        final String[] tokens = normalized.split("\\s+");
        final StringBuilder displayName = new StringBuilder();
        for (final String token : tokens) {
            if (token.isEmpty()) {
                continue;
            }
            if (displayName.length() > 0) {
                displayName.append(' ');
            }

            if (token.matches(".*\\d.*")) {
                displayName.append(token.toUpperCase(Locale.US));
            } else {
                displayName.append(Character.toUpperCase(token.charAt(0)));
                if (token.length() > 1) {
                    displayName.append(token.substring(1).toLowerCase(Locale.US));
                }
            }
        }

        return displayName.toString();
    }
}

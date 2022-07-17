package com.example.ece454finalproject;

import android.app.Activity;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.os.SystemClock;
import android.util.Log;

import org.tensorflow.lite.Interpreter;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;

class ImageClassification{

    static final int DIM_IMG_SIZE_X = 224;
    static final int DIM_IMG_SIZE_Y = 224;
    private static final String TAG = "ImageClassification";

    private static final String MODEL_PATH = "asl_model_v1.tflite";
    private static final String LABEL_PATH = "labels.txt";
    private static final int DIM_BATCH_SIZE = 1;
    private static final int DIM_PIXEL_SIZE = 3;

    /* Preallocated buffers for storing image data in. */
    private int[] intValues = new int[DIM_IMG_SIZE_X * DIM_IMG_SIZE_Y];

    private Interpreter tflite; // used to run inference
    private List<String> labelList; // used to store data in list.txt

    private ByteBuffer imgData = null; // buffer for input image data

    private byte[][] labelProbArray = null; // holds probabilities from inference

    /**
     * Initializes an ImageClassifier
     */
    public ImageClassification(Activity activity) throws IOException {
        tflite = new Interpreter(loadModelFile(activity));
        labelList = loadLabelList(activity);
        imgData =
                ByteBuffer.allocateDirect(
                        DIM_BATCH_SIZE * DIM_IMG_SIZE_X * DIM_IMG_SIZE_Y * DIM_PIXEL_SIZE);
        imgData.order(ByteOrder.nativeOrder());
        labelProbArray = new byte[1][labelList.size()];
        Log.d(TAG, "Created a Tensorflow Lite Image Classifier.");
    }

    String classifyFrame(Bitmap bitmap) {
        if (tflite == null) {
            Log.d(TAG, "Image classifier has not been initialized; Skipped.");
            return "Uninitialized Classifier.";
        }
        convertBitmapToByteBuffer(bitmap);
        // Here's where the magic happens!!!
        long startTime = SystemClock.uptimeMillis();
        tflite.run(imgData, labelProbArray);
        long endTime = SystemClock.uptimeMillis();
        Log.d(TAG, "Timecost to run model inference: " + Long.toString(endTime - startTime));
        String textToShow = getTopEntry();
        textToShow = Long.toString(endTime - startTime) + "ms" + textToShow;
        return textToShow;
    }

    /**
     * Reads label list from Assets.
     */
    private List<String> loadLabelList(Activity activity) throws IOException {
        List<String> labelList = new ArrayList<String>();
        BufferedReader reader =
                new BufferedReader(new InputStreamReader(activity.getAssets().open(LABEL_PATH)));
        String line;
        while ((line = reader.readLine()) != null) {
            labelList.add(line);
        }
        reader.close();
        return labelList;
    }

    /**
     * Memory-map the model file in Assets.
     */
    private MappedByteBuffer loadModelFile(Activity activity) throws IOException {
        AssetFileDescriptor fileDescriptor = activity.getAssets().openFd(MODEL_PATH);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    /**
     * Writes Image data into a {@code ByteBuffer}.
     */
    private void convertBitmapToByteBuffer(Bitmap bitmap) {
        if (imgData == null) {
            return;
        }
        imgData.rewind();
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        // Convert the image to floating point.
        int pixel = 0;
        long startTime = SystemClock.uptimeMillis();
        for (int i = 0; i < DIM_IMG_SIZE_X; ++i) {
            for (int j = 0; j < DIM_IMG_SIZE_Y; ++j) {
                final int val = intValues[pixel++];
                imgData.put((byte) ((val >> 16) & 0xFF));
                imgData.put((byte) ((val >> 8) & 0xFF));
                imgData.put((byte) (val & 0xFF));
            }
        }
        long endTime = SystemClock.uptimeMillis();
        Log.d(TAG, "Timecost to put values into ByteBuffer: " + Long.toString(endTime - startTime));
    }

    private String getTopEntry() {
        // iterate through labelProbArray, find highest probability and return it

        return null;
    }

    /**
     * Prints top-K labels, to be shown in UI as the results.
     */
//    private String printTopKLabels() {
//        for (int i = 0; i < labelList.size(); ++i) {
//            sortedLabels.add(
//                    new AbstractMap.SimpleEntry<>(labelList.get(i), (labelProbArray[0][i] & 0xff) / 255.0f));
//            if (sortedLabels.size() > RESULTS_TO_SHOW) {
//                sortedLabels.poll();
//            }
//        }
//        String textToShow = "";
//        final int size = sortedLabels.size();
//        for (int i = 0; i < size; ++i) {
//            Map.Entry<String, Float> label = sortedLabels.poll();
//            textToShow = "\n" + label.getKey() + ":" + Float.toString(label.getValue()) + textToShow;
//        }
//        return textToShow;
//    }

    public void close() {
        tflite.close();
        tflite = null;
    }
}
//package com.example.ece454finalproject;
//
//import android.content.Context;
//import android.graphics.Bitmap;
//import android.util.Log;
//
//import org.tensorflow.lite.support.image.TensorImage;
//import org.tensorflow.lite.task.core.BaseOptions;
//import org.tensorflow.lite.task.vision.classifier.Classifications;
//import org.tensorflow.lite.task.vision.classifier.ImageClassifier;
//
//import java.io.File;
//import java.io.IOException;
//import java.util.List;
//
//class ImageClassification {
//
//    private static final String TAG = "ImageClassification";
//    private static final String modelPath = "asl_model_v1.tflite";
//    private ImageClassifier imageClassifier;
//    private File modelFile = null;
//    private int NUM_RESULTS = 1;
//
//    // Initialization
//    private void initClassifier(Context context) {
//        // recommended options (referenced in github docs)
//        ImageClassifier.ImageClassifierOptions options =
//                ImageClassifier.ImageClassifierOptions.builder()
//                        .setBaseOptions(BaseOptions.builder().useGpu().build())
//                        .setMaxResults(NUM_RESULTS)
//                        .build();
//
//        try {
//            imageClassifier =
//                    ImageClassifier.createFromFileAndOptions(
//                            modelFile, options);
//        } catch (IOException e) {
//            e.printStackTrace();
//            Log.d(TAG, "Error creating image classifier model");
//        }
//        return;
//    }
//
//    // close ImageClassifier resource
//    private void closeClassifier() {
//        if (imageClassifier == null || imageClassifier.isClosed()) {
//            Log.d(TAG, "imageClassifier is null or already closed");
//            return;
//        }
//        imageClassifier.close();
//    }
//
//    // currently returns just one result, maybe return multiple during debugging
//    List<Classifications> classifyFrame(Bitmap bitmap) {
//        TensorImage tensorImage = null;
//        List<Classifications> result;
//        try {
//            result = imageClassifier.classify(tensorImage);
//            return result;
//        } catch (IllegalArgumentException e) {
//            Log.d(TAG, "Error classifying the image");
//            e.printStackTrace();
//        }
//        return null;
//    }
//
//    private void loadModelFile() {
//        modelFile = new File(modelPath);
//        return;
//    }
//}
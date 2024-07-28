// App.js
import React, { useState, useEffect, useRef } from 'react';
import { View, Text, TouchableOpacity, StyleSheet } from 'react-native';
import { Camera } from 'expo-camera';
import * as Speech from 'expo-speech';
import * as tf from '@tensorflow/tfjs';
import { cameraWithTensors } from '@tensorflow/tfjs-react-native';
import * as mobilenet from '@tensorflow-models/mobilenet';
import * as opencv from 'opencv-react-native';

const TensorCamera = cameraWithTensors(Camera);

const App = () => {
  const [hasPermission, setHasPermission] = useState(null);
  const [model, setModel] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [selectedLanguage, setSelectedLanguage] = useState('ASL');
  const cameraRef = useRef(null);

  useEffect(() => {
    (async () => {
      const { status } = await Camera.requestCameraPermissionsAsync();
      setHasPermission(status === 'granted');

      await tf.ready();
      const loadedModel = await mobilenet.load();
      setModel(loadedModel);
    })();
  }, []);

  const handleCameraStream = (images) => {
    const loop = async () => {
      const nextImageTensor = images.next().value;
      if (!nextImageTensor) return;

      const imageProcessed = await preprocessImage(nextImageTensor);
      const prediction = await model.classify(imageProcessed);

      setPrediction(prediction[0]);

      requestAnimationFrame(loop);
    };
    loop();
  };

  const preprocessImage = async (imageTensor) => {
    const imageMat = await opencv.matFromImageData(imageTensor);
    opencv.cvtColor(imageMat, imageMat, opencv.COLOR_RGBA2GRAY);
    opencv.GaussianBlur(imageMat, imageMat, { width: 5, height: 5 }, 0);
    opencv.Canny(imageMat, imageMat, 50, 150);
    return tf.browser.fromPixels(imageMat);
  };

  const speakPrediction = () => {
    if (prediction) {
      Speech.speak(prediction.className);
    }
  };

  if (hasPermission === null) {
    return <View />;
  }
  if (hasPermission === false) {
    return <Text>No access to camera</Text>;
  }

  return (
    <View style={styles.container}>
      <TensorCamera
        style={styles.camera}
        type={Camera.Constants.Type.back}
        ref={cameraRef}
        onReady={handleCameraStream}
        resizeHeight={224}
        resizeWidth={224}
        resizeDepth={3}
        autorender={true}
      />
      <View style={styles.predictionContainer}>
        <Text style={styles.predictionText}>
          {prediction ? prediction.className : 'No prediction yet'}
        </Text>
        <TouchableOpacity style={styles.speakButton} onPress={speakPrediction}>
          <Text>Speak</Text>
        </TouchableOpacity>
      </View>
      <View style={styles.languageSelector}>
        <TouchableOpacity
          style={[styles.languageButton, selectedLanguage === 'ASL' && styles.selectedLanguage]}
          onPress={() => setSelectedLanguage('ASL')}
        >
          <Text>ASL</Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={[styles.languageButton, selectedLanguage === 'BSL' && styles.selectedLanguage]}
          onPress={() => setSelectedLanguage('BSL')}
        >
          <Text>BSL</Text>
        </TouchableOpacity>
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  camera: {
    flex: 1,
  },
  predictionContainer: {
    position: 'absolute',
    bottom: 0,
    left: 0,
    right: 0,
    backgroundColor: 'rgba(0,0,0,0.7)',
    padding: 20,
  },
  predictionText: {
    color: 'white',
    fontSize: 18,
    textAlign: 'center',
  },
  speakButton: {
    backgroundColor: 'white',
    padding: 10,
    borderRadius: 5,
    marginTop: 10,
    alignItems: 'center',
  },
  languageSelector: {
    position: 'absolute',
    top: 40,
    right: 20,
    flexDirection: 'row',
  },
  languageButton: {
    backgroundColor: 'white',
    padding: 10,
    borderRadius: 5,
    marginLeft: 10,
  },
  selectedLanguage: {
    backgroundColor: 'yellow',
  },
});

export default App;

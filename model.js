// signLanguageModel.js
import * as tf from '@tensorflow/tfjs';

const createModel = () => {
  const model = tf.sequential();
  
  model.add(tf.layers.conv2d({
    inputShape: [224, 224, 3],
    kernelSize: 3,
    filters: 16,
    activation: 'relu'
  }));
  model.add(tf.layers.maxPooling2d({poolSize: [2, 2]}));
  
  model.add(tf.layers.conv2d({
    kernelSize: 3,
    filters: 32,
    activation: 'relu'
  }));
  model.add(tf.layers.maxPooling2d({poolSize: [2, 2]}));
  
  model.add(tf.layers.flatten());
  
  model.add(tf.layers.dense({
    units: 64,
    activation: 'relu'
  }));
  
  model.add(tf.layers.dense({
    units: 26,  // Assuming 26 classes for letters A-Z
    activation: 'softmax'
  }));

  return model;
};

const trainModel = async (model, trainData, trainLabels) => {
  const optimizer = tf.train.adam();
  model.compile({
    optimizer: optimizer,
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
  });

  const history = await model.fit(trainData, trainLabels, {
    epochs: 10,
    validationSplit: 0.2,
    callbacks: {
      onEpochEnd: (epoch, log) => console.log(`Epoch ${epoch}: loss = ${log.loss}`)
    }
  });

  return history;
};

export { createModel, trainModel };

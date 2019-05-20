/**
 * This script is adapted from the two tensorflowjs examples hosted at
 * https://github.com/tensorflow/tfjs-examples/tree/master/webcam-transfer-learning
 * and https://github.com/tensorflow/tfjs-models/tree/master/posenet/demos
 */

const webcam = document.getElementById("webcam");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

let model;
let modelURL;
let imageDims;
let canvasDims;
let modelChange;

/**
 * This function captures an image from the webcam, resizes it to the preferred
 * dimensions of the selected model, and represents it in a format suitable for
 * input to the network.
 */
function fetchInputImage() {
  return tf.tidy(() => {
    const webcamImage = tf.browser.fromPixels(webcam);

    const batchedImage = webcamImage.toFloat().expandDims();

    const resizedImage = tf.image.resizeBilinear(batchedImage, imageDims, true);

    const clippedImage = tf.clipByValue(resizedImage, 0.0, 255.0);

    const reversedImage = tf.reverse(clippedImage, 2);

    return reversedImage;
  });
}

/**
 * A webcam image is fed to the model and the output is resized to the
 * original webcam resolution.
 */
function predictSaliency() {
  return tf.tidy(() => {
    const modelOutput = model.predict(fetchInputImage());

    const resizedOutput = tf.image.resizeBilinear(modelOutput, canvasDims, true);

    const clippedOutput = tf.clipByValue(resizedOutput, 0.0, 255.0);

    return clippedOutput.squeeze();
  });
}

/**
 * Here the model is loaded and fed with an initial image to warm up the
 * graph execution such that the next prediction will run faster. Afterwards,
 * the network keeps on predicting saliency as long as no other model is
 * selected. The results are automatically drawn to the canvas.
 */
async function runModel() {
  showLoadingScreen();

  model = await tf.loadGraphModel(modelURL);

  tf.tidy(() => model.predict(fetchInputImage())); // warmup

  modelChange = false;

  while (!modelChange) {
    const saliencyMap = predictSaliency();

    await tf.browser.toPixels(saliencyMap, canvas);

    saliencyMap.dispose();

    await tf.nextFrame();
  }

  model.dispose();

  runModel();
}

/**
 * When a new model is currently loading, the canvas signals a message
 * to the user.
 */
function showLoadingScreen() {
  ctx.fillStyle = "white";
  ctx.textAlign = "center";
  ctx.font = "1.7em Alegreya Sans SC", "1.7em sans-serif";
  ctx.fillText("loading model...", canvas.width / 2, canvas.height / 2);
}

/**
 * If a webcam or camera is detected, it is initialized with the specified
 * width and height and the function returns a successful promise.
 */
async function setupWebcam() {
  if (navigator.mediaDevices.getUserMedia) {
    const stream = await navigator.mediaDevices.getUserMedia({
      "audio": false,
      "video": {
        width: {
          min: 640,
          max: 640
        },
        height: {
          min: 480,
          max: 480
        }
      }
    });

    webcam.srcObject = stream;

    return new Promise((resolve) => {
      webcam.onloadedmetadata = () => {
        webcam.width = stream.getVideoTracks()[0].getSettings().width;
        webcam.height = stream.getVideoTracks()[0].getSettings().height;
        canvas.width = stream.getVideoTracks()[0].getSettings().width;
        canvas.height = stream.getVideoTracks()[0].getSettings().height;

        canvasDims = [canvas.height, canvas.width];

        resolve(webcam);
      };
    });
  }
}

/**
 * The main function that first defines the default model, adds mouse click
 * listeners that interrupt the current prediction loop and invoke the loading
 * of a different model, and tries to set up a webcam stream for input to the
 * model.
 */
async function app() {
  modelURL = "https://storage.googleapis.com/msi-net/model/very_low/model.json";
  imageDims = [48, 64];

  document.getElementById("very_low").addEventListener("click", () => {
    modelURL = "https://storage.googleapis.com/msi-net/model/very_low/model.json";
    imageDims = [48, 64];
    modelChange = true;
  });

  document.getElementById("low").addEventListener("click", () => {
    modelURL = "https://storage.googleapis.com/msi-net/model/low/model.json";
    imageDims = [72, 96];
    modelChange = true;
  });

  document.getElementById("medium").addEventListener("click", () => {
    modelURL = "https://storage.googleapis.com/msi-net/model/medium/model.json";
    imageDims = [120, 160];
    modelChange = true;
  });

  document.getElementById("high").addEventListener("click", () => {
    modelURL = "https://storage.googleapis.com/msi-net/model/high/model.json";
    imageDims = [168, 224];
    modelChange = true;
  });

  document.getElementById("very_high").addEventListener("click", () => {
    modelURL = "https://storage.googleapis.com/msi-net/model/very_high/model.json";
    imageDims = [240, 320];
    modelChange = true;
  });

  const noWebcamError = document.getElementById("error");

  try {
    await setupWebcam();
    noWebcamError.style.display = "none";
  } catch (DOMException) {
    noWebcamError.style.visibility = "visible";
    return;
  }

  runModel();
}

app();
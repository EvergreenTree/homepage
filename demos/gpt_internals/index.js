import {buildModel, DEFAULT_GPT_CONFIG, createPositionTensor} from './model.js';
import {loadData} from './data.js';

const tf = window.tf;
const tfvis = window.tfvis;

let data = null;
let model = null;
let attentionProbe = null;
let gptConfig = {...DEFAULT_GPT_CONFIG};
let training = false;
let stopRequested = false;

const positionCache = new Map();

function getCachedPositions(batchSize) {
  const key = `${batchSize}x${gptConfig.blockSize}`;
  if (!positionCache.has(key)) {
    const tensor = createPositionTensor(batchSize, gptConfig.blockSize);
    tf.keep(tensor);
    positionCache.set(key, tensor);
  }
  return positionCache.get(key);
}

function setStatus(message) {
  const el = document.getElementById('training-status');
  if (el) {
    el.textContent = message;
  }
}

function formatCharDisplay(ch) {
  if (ch === '\\n') {
    return '↵';
  } else if (ch === ' ') {
    return '(space)';
  }
  return ch;
}

function renderDatasetStats() {
  const stats = data.stats();
  const table = {
    headers: ['Metric', 'Value'],
    values: [
      ['Corpus length (chars)', stats.textLength.toLocaleString()],
      ['Total tokens', stats.tokens.toLocaleString()],
      ['Train split', stats.trainTokens.toLocaleString()],
      ['Validation split', stats.valTokens.toLocaleString()],
      ['Block size', stats.blockSize],
      ['Vocab size', stats.vocabSize],
    ],
  };
  
  // Use visor surface instead of DOM element
  const surface = {name: 'Dataset Statistics', tab: 'Data'};
  tfvis.render.table(surface, table);

  const freq = data.charFrequencies().slice(0, 30).map(({char, count}) => ({
    index: formatCharDisplay(char),
    value: count,
  }));
  
  if (freq.length) {
    const chartSurface = {name: 'Character Frequency', tab: 'Data'};
    tfvis.render.barchart(chartSurface, freq, {
      xLabel: 'Character',
      yLabel: 'Occurrences',
      height: 320,
    });
  }
}

function showModelSummary() {
  if (!model) {
    return;
  }
  const surface = {name: 'GPT Model Summary', tab: 'Model'};
  tfvis.show.modelSummary(surface, model);
}

function oneHotTargets(tensor, vocabSize) {
  const [batch, seqLen] = tensor.shape;
  const flattened = tensor.reshape([-1]);
  const oneHot = tf.oneHot(flattened, vocabSize);
  return oneHot.reshape([batch, seqLen, vocabSize]);
}

function getTrainingParametersFromUI() {
  return {
    trainSize: parseInt(document.getElementById('train-size').value),
    valSize: parseInt(document.getElementById('val-size').value),
    batchSize: parseInt(document.getElementById('batch-size').value),
    epochs: parseInt(document.getElementById('epochs').value),
  };
}

function validateTrainingParameters(params) {
  const errors = [];
  
  if (params.trainSize < 512 || params.trainSize > 8192) {
    errors.push('Train size must be between 512 and 8192');
  }
  
  if (params.valSize < 128 || params.valSize > 2048) {
    errors.push('Validation size must be between 128 and 2048');
  }
  
  if (params.batchSize < 8 || params.batchSize > 128) {
    errors.push('Batch size must be between 8 and 128');
  }
  
  if (params.epochs < 1 || params.epochs > 100) {
    errors.push('Epochs must be between 1 and 100');
  }
  
  if (params.trainSize % params.batchSize !== 0) {
    errors.push('Train size must be divisible by batch size');
  }
  
  return errors;
}

async function trainModel() {
  if (!data || !model || training) {
    return;
  }

  const startButton = document.getElementById('start-training');
  const stopButton = document.getElementById('stop-training');
  startButton.disabled = true;
  stopButton.disabled = false;

  const trainingParams = getTrainingParametersFromUI();
  const errors = validateTrainingParameters(trainingParams);
  
  if (errors.length > 0) {
    setStatus(`Invalid training parameters: ${errors.join(', ')}`);
    startButton.disabled = false;
    stopButton.disabled = true;
    return;
  }
  
  const {trainSize, valSize, batchSize, epochs} = trainingParams;

  training = true;
  stopRequested = false;
  model.stopTraining = false;
  setStatus('Preparing batches…');

  const trainBatch = data.nextTrainBatch(trainSize, gptConfig.blockSize);
  const valBatch = data.nextValBatch(valSize, gptConfig.blockSize);

  const trainPositions = createPositionTensor(trainSize, gptConfig.blockSize);
  const valPositions = createPositionTensor(valSize, gptConfig.blockSize);

  const trainLabels = oneHotTargets(trainBatch.ys, data.vocabSize).toFloat();
  const valLabels = oneHotTargets(valBatch.ys, data.vocabSize).toFloat();

  const metrics = ['loss', 'val_loss'];
  const surface = {name: 'GPT Training', tab: 'Training'};
  const fitCallbacks = tfvis.show.fitCallbacks(surface, metrics, {callbacks: ['onBatchEnd', 'onEpochEnd']});

  const customCallback = {
    onEpochBegin: async (epoch) => {
      setStatus(`Epoch ${epoch + 1} / ${epochs}`);
      await tf.nextFrame();
    },
    onEpochEnd: async (epoch, logs) => {
      const perplexity = Math.exp(Math.min(10, logs.loss));
      const val = logs.val_loss != null ? logs.val_loss.toFixed(3) : '–';
      setStatus(
          `Epoch ${epoch + 1}/${epochs} — loss ${logs.loss.toFixed(3)} ` +
          `(val ${val}) · perplexity ≈ ${perplexity.toFixed(2)}`);
      await tf.nextFrame();
      if (stopRequested) {
        model.stopTraining = true;
      }
    },
    onTrainEnd: () => {
      training = false;
      startButton.disabled = false;
      stopButton.disabled = true;
      setStatus('Training complete. Try sampling some text!');
      tf.dispose([trainBatch.xs, trainBatch.ys, trainPositions, trainLabels,
        valBatch.xs, valBatch.ys, valPositions, valLabels]);
    }
  };

  try {
    await model.fit(
        {tokens: trainBatch.xs, positions: trainPositions},
        trainLabels,
        {
          batchSize,
          epochs,
          shuffle: true,
          validationData: [
            {tokens: valBatch.xs, positions: valPositions},
            valLabels,
          ],
          callbacks: [fitCallbacks, customCallback],
        });
  } catch (err) {
    console.error(err);
    setStatus(`Training interrupted: ${err.message}`);
    tf.dispose([trainBatch.xs, trainBatch.ys, trainPositions, trainLabels,
      valBatch.xs, valBatch.ys, valPositions, valLabels]);
    training = false;
    startButton.disabled = false;
    stopButton.disabled = true;
  }
}

function sampleFromDistribution(probs, temperature = 1.0) {
  const adjusted = new Float32Array(probs.length);
  let total = 0;
  for (let i = 0; i < probs.length; i++) {
    const logProb = Math.log(Math.max(probs[i], 1e-9));
    const scaled = Math.exp(logProb / temperature);
    adjusted[i] = scaled;
    total += scaled;
  }
  let r = Math.random() * total;
  for (let i = 0; i < adjusted.length; i++) {
    r -= adjusted[i];
    if (r <= 0) {
      return i;
    }
  }
  return adjusted.length - 1;
}

function buildPaddedContext(ids) {
  const blockSize = gptConfig.blockSize;
  const context = ids.slice(-blockSize);
  const padded = new Int32Array(blockSize);
  padded.fill(0);
  const offset = blockSize - context.length;
  for (let i = 0; i < context.length; i++) {
    padded[offset + i] = context[i];
  }
  return {padded, offset, length: context.length};
}

async function generateText() {
  if (!model || !data) {
    return;
  }
  const prompt = document.getElementById('prompt-text').value || 'Friends, Romans, countrymen';
  const length = Math.max(1, Number(document.getElementById('generate-length').value) || 80);
  const temperature = Math.max(0.1, Number(document.getElementById('temperature').value) || 1.0);

  let ids = data.encode(prompt);
  let text = prompt;
  let lastProbs = null;
  let lastContext = null;

  for (let i = 0; i < length; i++) {
    const {padded, offset} = buildPaddedContext(ids);
    const tokensTensor = tf.tensor2d(padded, [1, gptConfig.blockSize], 'int32');
    const posTensor = getCachedPositions(1);

    const preds = tf.tidy(() => model.predict([tokensTensor, posTensor]));
    const lastStep = preds.slice([0, gptConfig.blockSize - 1, 0], [1, 1, -1]).reshape([data.vocabSize]);
    const probsTensor = tf.tidy(() => lastStep.div(lastStep.sum()));
    const probsData = await probsTensor.data();

    preds.dispose();
    lastStep.dispose();
    probsTensor.dispose();
    tokensTensor.dispose();

    lastProbs = Array.from(probsData);
    lastContext = {padded, offset};

    const nextId = sampleFromDistribution(lastProbs, temperature);
    ids.push(nextId);
    text += data.decode([nextId]);
    await tf.nextFrame();
  }

  document.getElementById('sample-output').textContent = text;
  if (lastProbs && lastContext) {
    renderNextTokenChart(lastProbs);
    renderAttentionFromContext(lastContext);
  }
}

function renderNextTokenChart(probs) {
  const ranked = probs.map((value, id) => ({
    id,
    char: data.decode([id]),
    value,
  }));
  ranked.sort((a, b) => b.value - a.value);
  const top = ranked.slice(0, 20).map(d => ({
    index: formatCharDisplay(d.char),
    value: d.value,
  }));
  const surface = {name: 'Next Token Probabilities', tab: 'Generation'};
  tfvis.render.barchart(surface, top, {
    xLabel: 'Character',
    yLabel: 'Probability',
    height: 320,
  });
}

async function renderAttentionFromContext(contextInfo) {
  if (!attentionProbe) {
    return;
  }
  const {padded, offset} = contextInfo;
  const tokensTensor = tf.tensor2d(padded, [1, gptConfig.blockSize], 'int32');
  const posTensor = getCachedPositions(1);
  const attnTensor = tf.tidy(() => attentionProbe.predict([tokensTensor, posTensor]));
  // attnTensor shape: [1, numHeads, blockSize, blockSize]
  // Remove batch dimension first, then average over heads
  const squeezed = attnTensor.squeeze([0]); // Remove batch dimension: [numHeads, blockSize, blockSize]
  const meanHeads = squeezed.mean(0); // Average over heads: [blockSize, blockSize]
  const matrix = await meanHeads.array();

  attnTensor.dispose();
  squeezed.dispose();
  meanHeads.dispose();
  tokensTensor.dispose();

  const tokens = [];
  for (let i = 0; i < padded.length; i++) {
    const label = i < offset ? '(pad)' : formatCharDisplay(data.decode([padded[i]]));
    tokens.push(label);
  }

  const start = Math.max(0, offset);
  const trimmedValues = matrix.slice(start).map(row => row.slice(start));
  const trimmedTokens = tokens.slice(start);

  const container = document.getElementById('attention-surface');
  container.innerHTML = '';
  if (!trimmedTokens.length) {
    container.textContent = 'Provide a longer prompt to view attention weights.';
    return;
  }
  await tfvis.render.heatmap({name: 'Attention Weights', tab: 'Attention'}, {
    values: trimmedValues,
    xTickLabels: trimmedTokens,
    yTickLabels: trimmedTokens,
  }, {
    colorMap: 'blues',
    height: 360,
  });
}

async function analyzePromptAttention() {
  if (!data || !attentionProbe) {
    return;
  }
  const prompt = document.getElementById('prompt-text').value;
  const ids = data.encode(prompt);
  const context = buildPaddedContext(ids);
  await renderAttentionFromContext(context);
}

async function initData() {
  const button = document.getElementById('load-data');
  button.disabled = true;
  setStatus('Loading corpus…');
  try {
    data = await loadData({blockSize: gptConfig.blockSize});
    renderDatasetStats();
    setStatus('Data ready. Initialize the model next.');
    document.getElementById('init-model').disabled = false;
  } catch (err) {
    console.error(err);
    setStatus(`Failed to load data: ${err.message}`);
    button.disabled = false;
  }
}

function getHyperparametersFromUI() {
  return {
    blockSize: parseInt(document.getElementById('block-size').value),
    embeddingSize: parseInt(document.getElementById('embedding-size').value),
    numHeads: parseInt(document.getElementById('num-heads').value),
    nLayers: parseInt(document.getElementById('num-layers').value),
    ffHiddenSize: parseInt(document.getElementById('ff-hidden-size').value),
    learningRate: parseFloat(document.getElementById('learning-rate').value),
  };
}

function validateHyperparameters(params) {
  const errors = [];
  
  if (params.embeddingSize % params.numHeads !== 0) {
    errors.push('Embedding size must be divisible by number of heads');
  }
  
  if (params.blockSize < 16 || params.blockSize > 256) {
    errors.push('Block size must be between 16 and 256');
  }
  
  if (params.embeddingSize < 32 || params.embeddingSize > 512) {
    errors.push('Embedding size must be between 32 and 512');
  }
  
  if (params.numHeads < 1 || params.numHeads > 16) {
    errors.push('Number of heads must be between 1 and 16');
  }
  
  if (params.nLayers < 1 || params.nLayers > 8) {
    errors.push('Number of layers must be between 1 and 8');
  }
  
  if (params.learningRate < 0.0001 || params.learningRate > 0.01) {
    errors.push('Learning rate must be between 0.0001 and 0.01');
  }
  
  return errors;
}

function initModel() {
  if (!data) {
    return;
  }
  
  const uiParams = getHyperparametersFromUI();
  const errors = validateHyperparameters(uiParams);
  
  if (errors.length > 0) {
    setStatus(`Invalid parameters: ${errors.join(', ')}`);
    return;
  }
  
  const current = buildModel({
    vocabSize: data.vocabSize,
    ...uiParams
  });
  model = current.model;
  attentionProbe = current.attentionProbe;
  gptConfig = {...gptConfig, ...current.config};
  setStatus(`Model initialized with ${uiParams.nLayers} layers, ${uiParams.embeddingSize}d embeddings. Ready to train!`);
  document.getElementById('start-training').disabled = false;
  document.getElementById('generate-button').disabled = false;
  document.getElementById('analyze-attention').disabled = false;
  document.getElementById('show-model').disabled = false;
}

function setupListeners() {
  document.getElementById('load-data').addEventListener('click', initData);
  document.getElementById('init-model').addEventListener('click', initModel);
  document.getElementById('start-training').addEventListener('click', trainModel);
  document.getElementById('stop-training').addEventListener('click', () => {
    stopRequested = true;
    setStatus('Stop requested. Finishing current epoch…');
  });
  document.getElementById('generate-button').addEventListener('click', generateText);
  document.getElementById('analyze-attention').addEventListener('click', analyzePromptAttention);
  document.getElementById('show-model').addEventListener('click', showModelSummary);
  document.getElementById('show-visor').addEventListener('click', () => {
    try {
      tfvis.visor().open();
    } catch (err) {
      console.error(err);
    }
  });
}

document.addEventListener('DOMContentLoaded', () => {
  setupListeners();
  setStatus('Load the dataset to get started.');
});

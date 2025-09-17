import {buildModel, createPositionTensor} from './model.js';
import {loadData} from './data.js';

const tf = window.tf;
const tfvis = window.tfvis;

let data = null;
let reluModel = null;
let coluModel = null;
let rcoluModel = null;
let training = false;
let stopRequested = false;
const histories = {
  relu: {train: [], val: []},
  colu: {train: [], val: []},
  rcolu: {train: [], val: []},
};

let sharedTraining = null;
let currentConfig = null;
let combinedVisible = false;

function getElement(id) {
  const el = document.getElementById(id);
  if (!el) {
    throw new Error(`Missing element with id="${id}"`);
  }
  return el;
}

function getControls() {
  return {
    trainRelu: getElement('train-relu'),
    trainColu: getElement('train-colu'),
    trainRcolu: getElement('train-rcolu'),
    gatherCurves: getElement('gather-curves'),
    autoRun: getElement('auto-run'),
    stop: getElement('stop-training'),
  };
}

function setStatus(message) {
  const el = getElement('training-status');
  el.textContent = message;
}

function clearHistories() {
  histories.relu.train = [];
  histories.relu.val = [];
  histories.colu.train = [];
  histories.colu.val = [];
  histories.rcolu.train = [];
  histories.rcolu.val = [];
  renderAllVariants();
}

function resetHistoryFor(key) {
  histories[key].train = [];
  histories[key].val = [];
  renderVariantCurves(key);
  renderCombinedCurves();
}

function renderDatasetStats() {
  if (!data) {
    return;
  }
  const stats = data.stats();
  const target = getElement('dataset-stats');
  target.textContent = `Loaded corpus with ${stats.tokens.toLocaleString()} tokens (train ${stats.trainTokens.toLocaleString()}, val ${stats.valTokens.toLocaleString()})`;
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

  const surface = {name: 'Dataset Statistics', tab: 'Data'};
  tfvis.render.table(surface, table);
}

function renderPlaceholder(surface) {
  tfvis.render.linechart(surface, {values: [[{x: 0, y: 0}]], series: ['placeholder']}, {
    xLabel: 'Step',
    yLabel: 'Loss',
    width: 500,
    height: 360,
    seriesColors: ['rgba(0,0,0,0)'],
    yAxisDomain: [0, 1],
  });
}

function sanitizeHistorySeries(history) {
  if (!Array.isArray(history) || history.length === 0) {
    return [];
  }

  const sanitized = [];
  for (const point of history) {
    if (!point) {
      continue;
    }
    const step = Number(point.step);
    const loss = Number(point.loss);
    if (!Number.isFinite(step) || !Number.isFinite(loss)) {
      continue;
    }
    sanitized.push({x: step, y: loss});
  }

  if (sanitized.length > 1) {
    sanitized.sort((a, b) => a.x - b.x);
  }

  return sanitized;
}

function alignSeriesEntries(entries) {
  if (!entries.length) {
    return entries;
  }

  if (entries.length === 1) {
    const {label, points} = entries[0];
    return [{label, points: points.map(point => ({x: point.x, y: point.y}))}];
  }

  const reference = entries.reduce((longest, entry) => {
    if (!longest || entry.points.length > longest.points.length) {
      return entry;
    }
    return longest;
  }, null);

  const refPoints = reference?.points || [];
  if (!refPoints.length) {
    return entries.map(({label}) => ({label, points: []}));
  }

  return entries.map((entry) => {
    if (entry === reference) {
      return {
        label: entry.label,
        points: refPoints.map(point => ({x: point.x, y: point.y})),
      };
    }

    const srcPoints = entry.points;
    if (!srcPoints.length) {
      // No data yet—mirror reference steps with zeros to avoid tfvis errors.
      return {
        label: entry.label,
        points: refPoints.map(({x}) => ({x, y: 0})),
      };
    }

    const aligned = [];
    let srcIdx = 0;
    let lastY = srcPoints[0].y;

    for (const refPoint of refPoints) {
      while (srcIdx < srcPoints.length && srcPoints[srcIdx].x <= refPoint.x) {
        lastY = srcPoints[srcIdx].y;
        srcIdx += 1;
      }
      aligned.push({x: refPoint.x, y: lastY});
    }

    return {label: entry.label, points: aligned};
  });
}

const VARIANT_META = {
  relu: {
    surface: {name: 'ReLU Curve', tab: 'Training'},
    trainLabel: 'ReLU · train',
    valLabel: 'ReLU · val',
    colors: ['#1f77b4', 'rgba(31,119,180,0.35)'],
  },
  colu: {
    surface: {name: 'CoLU Curve', tab: 'Training'},
    trainLabel: 'CoLU · train',
    valLabel: 'CoLU · val',
    colors: ['#d62728', 'rgba(214,39,40,0.35)'],
  },
  rcolu: {
    surface: {name: 'RCoLU Curve', tab: 'Training'},
    trainLabel: 'RCoLU · train',
    valLabel: 'RCoLU · val',
    colors: ['#ff7f0e', 'rgba(255,127,14,0.35)'],
  },
};

function renderVariantCurves(variant) {
  const meta = VARIANT_META[variant];
  if (!meta) {
    return;
  }

  const entries = [
    {label: meta.trainLabel, history: histories[variant].train},
    {label: meta.valLabel, history: histories[variant].val},
  ];

  const sanitizedEntries = entries.map(({label, history}) => {
    const points = sanitizeHistorySeries(history);
    return points.length ? {label, points} : null;
  }).filter(Boolean);

  if (!sanitizedEntries.length) {
    renderPlaceholder(meta.surface);
    return;
  }

  const alignedEntries = alignSeriesEntries(sanitizedEntries);
  const series = alignedEntries.map(({label}) => label);
  const values = alignedEntries.map(({points}) => points);

  const colorMap = {
    [meta.trainLabel]: meta.colors[0],
    [meta.valLabel]: meta.colors[1] || meta.colors[0],
  };

  const renderResult = tfvis.render.linechart(meta.surface, {values, series}, {
    xLabel: 'Step',
    yLabel: 'Loss',
    width: 500,
    height: 360,
    seriesColors: series.map(label => colorMap[label] || '#444'),
  });

  if (renderResult && typeof renderResult.catch === 'function') {
    renderResult.catch((err) => {
      console.error('Error rendering variant curves:', err);
      console.log('Debug info:', {variant, values, series});
      renderPlaceholder(meta.surface);
    });
  }
}

function renderCombinedCurves() {
  if (!combinedVisible) {
    return;
  }
  const surface = {name: 'Combined Curves', tab: 'Training'};
  const entries = [
    {label: 'ReLU · train', history: histories.relu.train},
    {label: 'ReLU · val', history: histories.relu.val},
    {label: 'CoLU · train', history: histories.colu.train},
    {label: 'CoLU · val', history: histories.colu.val},
    {label: 'RCoLU · train', history: histories.rcolu.train},
    {label: 'RCoLU · val', history: histories.rcolu.val},
  ];

  const sanitizedEntries = entries.map(({label, history}) => {
    const points = sanitizeHistorySeries(history);
    return points.length ? {label, points} : null;
  }).filter(Boolean);

  if (!sanitizedEntries.length) {
    renderPlaceholder(surface);
    return;
  }

  const alignedEntries = alignSeriesEntries(sanitizedEntries);
  const series = alignedEntries.map(({label}) => label);
  const values = alignedEntries.map(({points}) => points);

  const colorMap = {
    'ReLU · train': '#1f77b4',
    'ReLU · val': 'rgba(31,119,180,0.35)',
    'CoLU · train': '#d62728',
    'CoLU · val': 'rgba(214,39,40,0.35)',
    'RCoLU · train': '#ff7f0e',
    'RCoLU · val': 'rgba(255,127,14,0.35)',
  };

  const renderResult = tfvis.render.linechart(surface, {values, series}, {
    xLabel: 'Step',
    yLabel: 'Loss',
    width: 500,
    height: 360,
    seriesColors: series.map(label => colorMap[label] || '#444'),
  });

  if (renderResult && typeof renderResult.catch === 'function') {
    renderResult.catch((err) => {
      console.error('Error rendering combined curves:', err);
      console.log('Debug info:', {values, series});
      renderPlaceholder(surface);
    });
  }
}

function renderAllVariants() {
  renderVariantCurves('relu');
  renderVariantCurves('colu');
  renderVariantCurves('rcolu');
  renderCombinedCurves();
}

function disposeSharedTraining() {
  if (!sharedTraining) {
    return;
  }
  sharedTraining.trainXs?.dispose();
  sharedTraining.trainLabels?.dispose();
  sharedTraining.trainPositions?.dispose();
  sharedTraining.valXs?.dispose();
  sharedTraining.valLabels?.dispose();
  sharedTraining.valPositions?.dispose();
  sharedTraining = null;
}

function oneHotTargets(tensor, vocabSize) {
  const [batch, seqLen] = tensor.shape;
  const flattened = tensor.reshape([-1]);
  const oneHot = tf.oneHot(flattened, vocabSize);
  return oneHot.reshape([batch, seqLen, vocabSize]);
}

function prepareSharedTraining(config) {
  const trainBatch = data.nextTrainBatch(config.trainSize, config.blockSize);
  const valBatch = data.nextValBatch(config.valSize, config.blockSize);

  const trainPositions = createPositionTensor(config.trainSize, config.blockSize);
  const valPositions = createPositionTensor(config.valSize, config.blockSize);

  const trainLabels = oneHotTargets(trainBatch.ys, data.vocabSize).toFloat();
  const valLabels = oneHotTargets(valBatch.ys, data.vocabSize).toFloat();

  trainBatch.ys.dispose();
  valBatch.ys.dispose();

  return {
    trainXs: trainBatch.xs,
    trainLabels,
    trainPositions,
    valXs: valBatch.xs,
    valLabels,
    valPositions,
  };
}

const CONFIG_KEYS = ['blockSize', 'embeddingSize', 'numHeads', 'nLayers', 'ffHiddenSize',
  'learningRate', 'trainSize', 'valSize', 'batchSize', 'epochs', 'coluDim'];

function configsEqual(a, b) {
  if (!a || !b) {
    return false;
  }
  return CONFIG_KEYS.every((key) => a[key] === b[key]);
}

function ensureSharedTraining(config) {
  if (!sharedTraining || !currentConfig || !configsEqual(currentConfig, config)) {
    disposeSharedTraining();
    data.blockSize = config.blockSize;
    sharedTraining = prepareSharedTraining(config);
    currentConfig = {...config};
    clearHistories();
  }
}

function parseConfigFromUI() {
  const blockSize = parseInt(getElement('block-size').value, 10);
  const embeddingSize = parseInt(getElement('embedding-size').value, 10);
  const numHeads = parseInt(getElement('num-heads').value, 10);
  const nLayers = parseInt(getElement('num-layers').value, 10);
  const ffHiddenSize = parseInt(getElement('ff-hidden-size').value, 10);
  const learningRate = parseFloat(getElement('learning-rate').value);
  const trainSize = parseInt(getElement('train-size').value, 10);
  const valSize = parseInt(getElement('val-size').value, 10);
  const batchSize = parseInt(getElement('batch-size').value, 10);
  const epochs = parseInt(getElement('epochs').value, 10);
  const coluDim = parseInt(getElement('colu-dim').value, 10);

  return {
    blockSize,
    embeddingSize,
    numHeads,
    nLayers,
    ffHiddenSize,
    learningRate,
    trainSize,
    valSize,
    batchSize,
    epochs,
    coluDim,
  };
}

function validateConfig(config) {
  const errors = [];
  if (config.trainSize < config.batchSize || config.trainSize % config.batchSize !== 0) {
    errors.push('Train size must be >= batch size and divisible by it');
  }
  if (config.valSize < config.batchSize) {
    errors.push('Validation size must be >= batch size');
  }
  if (config.ffHiddenSize % config.coluDim !== 0) {
    errors.push('FF hidden size must be divisible by CoLU dim');
  }
  if (config.blockSize <= 0) {
    errors.push('Block size must be positive');
  }
  if (config.coluDim < 2) {
    errors.push('CoLU dim must be at least 2');
  }
  return errors;
}

function disposeModels() {
  if (reluModel) {
    reluModel.dispose();
    reluModel = null;
  }
  if (coluModel) {
    coluModel.dispose();
    coluModel = null;
  }
  if (rcoluModel) {
    rcoluModel.dispose();
    rcoluModel = null;
  }
}

async function trainVariant(model, key, config, shared) {
  const {trainXs, trainLabels, trainPositions, valXs, valLabels, valPositions} = shared;
  const {epochs, batchSize} = config;
  const displayName = key === 'relu' ? 'ReLU' : key === 'colu' ? 'CoLU' : 'RCoLU';
  let step = 0;

  return model.fit(
      {tokens: trainXs, positions: trainPositions},
      trainLabels,
      {
        batchSize,
        epochs,
        shuffle: true,
        validationData: [
          {tokens: valXs, positions: valPositions},
          valLabels,
        ],
        callbacks: {
          onEpochBegin: async (epoch) => {
            setStatus(`[${displayName}] Epoch ${epoch + 1} / ${epochs}`);
            await tf.nextFrame();
          },
          onBatchEnd: async (batch, logs) => {
            step += 1;
            if (logs.loss != null && typeof logs.loss === 'number' && !isNaN(logs.loss)) {
              histories[key].train.push({step, loss: logs.loss});
            }
            renderVariantCurves(key);
            renderCombinedCurves();
            await tf.nextFrame();
            if (stopRequested) {
              model.stopTraining = true;
            }
          },
          onEpochEnd: async (epoch, logs) => {
            if (logs.val_loss != null && typeof logs.val_loss === 'number' && !isNaN(logs.val_loss)) {
              const valStep = step;
              histories[key].val.push({step: valStep, loss: logs.val_loss});
            }
            renderVariantCurves(key);
            renderCombinedCurves();
            await tf.nextFrame();
            if (stopRequested) {
              model.stopTraining = true;
            }
          },
        },
      });
}

async function trainActivation(variant) {
  if (training) {
    return;
  }

  if (!data) {
    setStatus('Load the dataset before training.');
    return;
  }

  const config = parseConfigFromUI();
  const errors = validateConfig(config);
  if (errors.length) {
    setStatus(`Invalid configuration: ${errors.join(', ')}`);
    return;
  }

  try {
    ensureSharedTraining(config);
  } catch (err) {
    console.error(err);
    disposeSharedTraining();
    currentConfig = null;
    setStatus(`Failed to prepare shared batches: ${err.message}`);
    return;
  }

  if (!sharedTraining) {
    setStatus('Failed to prepare shared batches.');
    return;
  }

  const controls = getControls();
  controls.trainRelu.disabled = true;
  controls.trainColu.disabled = true;
  controls.trainRcolu.disabled = true;
  controls.gatherCurves.disabled = true;
  controls.autoRun.disabled = true;
  controls.stop.disabled = false;

  training = true;
  stopRequested = false;

  const displayName = variant === 'relu' ? 'ReLU' : variant === 'colu' ? 'CoLU' : 'RCoLU';

  try {
    if (variant === 'relu' && reluModel) {
      reluModel.dispose();
      reluModel = null;
    }
    if (variant === 'colu' && coluModel) {
      coluModel.dispose();
      coluModel = null;
    }
    if (variant === 'rcolu' && rcoluModel) {
      rcoluModel.dispose();
      rcoluModel = null;
    }

    resetHistoryFor(variant);

    const modelConfig = {
      ...currentConfig,
      vocabSize: data.vocabSize,
      activation: variant,
    };
    if (variant === 'colu' || variant === 'rcolu') {
      modelConfig.coluDim = currentConfig.coluDim;
    }

    const model = buildModel(modelConfig);
    if (variant === 'relu') {
      reluModel = model;
    } else if (variant === 'colu') {
      coluModel = model;
    } else if (variant === 'rcolu') {
      rcoluModel = model;
    }

    setStatus(`${displayName} training…`);
    await tf.nextFrame();

    await trainVariant(model, variant, currentConfig, sharedTraining);
    const wasStopped = stopRequested;
    setStatus(wasStopped ? `${displayName} training stopped` : `${displayName} training complete`);
  } catch (err) {
    console.error(err);
    setStatus(`${displayName} training error: ${err.message}`);
  } finally {
    training = false;
    stopRequested = false;
    controls.stop.disabled = true;
    controls.trainRelu.disabled = false;
    controls.trainColu.disabled = false;
    controls.trainRcolu.disabled = false;
    controls.gatherCurves.disabled = false;
    controls.autoRun.disabled = false;
    renderVariantCurves(variant);
    renderCombinedCurves();
  }
}

function gatherCurves() {
  const wasVisible = combinedVisible;
  combinedVisible = true;
  renderCombinedCurves();
  if (!wasVisible) {
    if (!histories.relu.train.length && !histories.colu.train.length && !histories.rcolu.train.length) {
      setStatus('Combined chart ready. Train any model to populate curves.');
    } else {
      setStatus('Combined chart displayed.');
    }
  } else {
    setStatus('Combined chart refreshed.');
  }
}

function requestStop() {
  if (!training) {
    return;
  }
  if (!stopRequested) {
    stopRequested = true;
    setStatus('Stop requested… waiting for the next batch.');
  }
}

async function autoRunAll() {
  if (training) {
    return;
  }

  if (!data) {
    setStatus('Load the dataset before running automation.');
    return;
  }

  const controls = getControls();
  
  // Disable all controls during automation
  controls.trainRelu.disabled = true;
  controls.trainColu.disabled = true;
  controls.trainRcolu.disabled = true;
  controls.gatherCurves.disabled = true;
  controls.autoRun.disabled = true;
  controls.stop.disabled = false;

  try {
    setStatus('Starting automated training sequence...');
    
    // Train ReLU
    if (!stopRequested) {
      setStatus('Auto-training ReLU...');
      await trainActivation('relu');
      await new Promise(resolve => setTimeout(resolve, 1000)); // Brief pause
    }
    
    // Train CoLU
    if (!stopRequested) {
      setStatus('Auto-training CoLU...');
      await trainActivation('colu');
      await new Promise(resolve => setTimeout(resolve, 1000)); // Brief pause
    }
    
    // Train RCoLU
    if (!stopRequested) {
      setStatus('Auto-training RCoLU...');
      await trainActivation('rcolu');
      await new Promise(resolve => setTimeout(resolve, 1000)); // Brief pause
    }
    
    // Gather curves
    if (!stopRequested) {
      setStatus('Gathering comparison curves...');
      gatherCurves();
      setStatus('Automated training sequence complete!');
    } else {
      setStatus('Automated training sequence stopped.');
    }
    
  } catch (err) {
    console.error('Auto-run error:', err);
    setStatus(`Auto-run error: ${err.message}`);
  } finally {
    // Re-enable controls
    controls.trainRelu.disabled = false;
    controls.trainColu.disabled = false;
    controls.trainRcolu.disabled = false;
    controls.gatherCurves.disabled = false;
    controls.autoRun.disabled = false;
    controls.stop.disabled = true;
  }
}

function hookControls() {
  const controls = getControls();

  getElement('show-visor').addEventListener('click', () => {
    const visor = tfvis.visor();
    if (!visor.isOpen()) {
      visor.open();
    } else {
      visor.close();
    }
  });

  getElement('load-data').addEventListener('click', async () => {
    if (training) {
      return;
    }
    setStatus('Loading dataset…');
    controls.trainRelu.disabled = true;
    controls.trainColu.disabled = true;
    controls.trainRcolu.disabled = true;
    controls.gatherCurves.disabled = true;
    controls.autoRun.disabled = true;
    controls.stop.disabled = true;
    try {
      combinedVisible = false;
      disposeSharedTraining();
      disposeModels();
      clearHistories();
      currentConfig = null;
      renderPlaceholder({name: 'Combined Curves', tab: 'Training'});

      data = await loadData({blockSize: parseInt(getElement('block-size').value, 10)});
      renderDatasetStats();
      setStatus('Dataset ready. Train ReLU, CoLU, or RCoLU to begin.');
      controls.trainRelu.disabled = false;
      controls.trainColu.disabled = false;
      controls.trainRcolu.disabled = false;
      controls.gatherCurves.disabled = false;
      controls.autoRun.disabled = false;
    } catch (err) {
      console.error(err);
      setStatus(`Failed to load data: ${err.message}`);
    }
  });

  controls.trainRelu.addEventListener('click', () => trainActivation('relu'));
  controls.trainColu.addEventListener('click', () => trainActivation('colu'));
  controls.trainRcolu.addEventListener('click', () => trainActivation('rcolu'));
  controls.autoRun.addEventListener('click', autoRunAll);
  controls.stop.addEventListener('click', requestStop);
  controls.gatherCurves.addEventListener('click', gatherCurves);
}

function init() {
  hookControls();
  combinedVisible = false;
  renderAllVariants();
  const controls = getControls();
  controls.trainRelu.disabled = true;
  controls.trainColu.disabled = true;
  controls.trainRcolu.disabled = true;
  controls.gatherCurves.disabled = true;
  controls.autoRun.disabled = true;
  controls.stop.disabled = true;
  setStatus('Load the dataset to begin.');
}

window.addEventListener('load', init);

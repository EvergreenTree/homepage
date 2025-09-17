const tf = window.tf;

const MASK_CACHE = new Map();

export const DEFAULT_GPT_CONFIG = {
  blockSize: 64,
  embeddingSize: 64,
  numHeads: 4,
  nLayers: 3,
  ffHiddenSize: 256,
  learningRate: 5e-3,
  activation: 'relu',
  coluDim: 4,
  coluEps: 1e-7,
};

function getCausalMask(length) {
  if (!MASK_CACHE.has(length)) {
    const buffer = tf.buffer([length, length], 'float32');
    for (let i = 0; i < length; i++) {
      for (let j = 0; j < length; j++) {
        buffer.set(j <= i ? 0 : -1e9, i, j);
      }
    }
    const mask = buffer.toTensor();
    tf.keep(mask);
    MASK_CACHE.set(length, mask);
  }
  return MASK_CACHE.get(length);
}

class CausalSelfAttention extends tf.layers.Layer {
  static get className() {
    return 'CausalSelfAttention';
  }

  constructor(config) {
    super(config);
    this.supportsMasking = false;
    this.numHeads = config.numHeads || 4;
    this.embedDim = config.embedDim;
    this.blockSize = config.blockSize;
    this.headDim = Math.floor(this.embedDim / this.numHeads);
    if (this.headDim * this.numHeads !== this.embedDim) {
      throw new Error(`Embedding size ${this.embedDim} must be divisible by numHeads ${this.numHeads}`);
    }
  }

  build(inputShape) {
    const kernelInit = tf.initializers.glorotUniform({seed: 42});
    const biasInit = tf.initializers.zeros();
    this.wq = this.addWeight('wq', [this.embedDim, this.embedDim], 'float32', kernelInit);
    this.wk = this.addWeight('wk', [this.embedDim, this.embedDim], 'float32', kernelInit);
    this.wv = this.addWeight('wv', [this.embedDim, this.embedDim], 'float32', kernelInit);
    this.wo = this.addWeight('wo', [this.embedDim, this.embedDim], 'float32', kernelInit);

    this.bq = this.addWeight('bq', [this.embedDim], 'float32', biasInit);
    this.bk = this.addWeight('bk', [this.embedDim], 'float32', biasInit);
    this.bv = this.addWeight('bv', [this.embedDim], 'float32', biasInit);
    this.bo = this.addWeight('bo', [this.embedDim], 'float32', biasInit);

    super.build(inputShape);
  }

  call(inputs, kwargs) {
    return tf.tidy(() => {
      const x = Array.isArray(inputs) ? inputs[0] : inputs;
      const batchSize = x.shape[0];
      const seqLen = x.shape[1];

      const flat = x.reshape([-1, this.embedDim]);
      const q = flat.matMul(this.wq.read()).add(this.bq.read());
      const k = flat.matMul(this.wk.read()).add(this.bk.read());
      const v = flat.matMul(this.wv.read()).add(this.bv.read());

      const heads = this.numHeads;
      const headDim = this.headDim;
      const reshapeHeads = tensor => tensor.reshape([batchSize, seqLen, heads, headDim])
                                       .transpose([0, 2, 1, 3])
                                       .reshape([batchSize * heads, seqLen, headDim]);

      const qH = reshapeHeads(q);
      const kH = reshapeHeads(k);
      const vH = reshapeHeads(v);

      let scores = tf.matMul(qH, kH, false, true);
      const scale = 1 / Math.sqrt(headDim);
      scores = scores.mul(scale);

      const mask = getCausalMask(seqLen).expandDims(0).tile([batchSize * heads, 1, 1]);
      scores = scores.add(mask);

      const attn = tf.softmax(scores, -1);
      const context = tf.matMul(attn, vH);

      const contextReshaped = context.reshape([batchSize, heads, seqLen, headDim])
                                   .transpose([0, 2, 1, 3])
                                   .reshape([batchSize * seqLen, this.embedDim]);

      const out = contextReshaped.matMul(this.wo.read()).add(this.bo.read())
                       .reshape([batchSize, seqLen, this.embedDim]);

      return out;
    });
  }

  getConfig() {
    const baseConfig = super.getConfig();
    return Object.assign({}, baseConfig, {
      numHeads: this.numHeads,
      embedDim: this.embedDim,
      blockSize: this.blockSize,
    });
  }
}

tf.serialization.registerClass(CausalSelfAttention);

class ColuActivation extends tf.layers.Layer {
  static get className() {
    return 'ColuActivation';
  }

  constructor(config) {
    super(config);
    this.dim = config.dim ?? DEFAULT_GPT_CONFIG.coluDim;
    this.eps = config.eps ?? DEFAULT_GPT_CONFIG.coluEps;
  }

  call(inputs) {
    return tf.tidy(() => {
      const x = Array.isArray(inputs) ? inputs[0] : inputs;
      return coluActivationTensor(x, this.dim, this.eps);
    });
  }

  getConfig() {
    const baseConfig = super.getConfig();
    return Object.assign({}, baseConfig, {
      dim: this.dim,
      eps: this.eps,
    });
  }
}

tf.serialization.registerClass(ColuActivation);

class RcoluActivation extends tf.layers.Layer {
  static get className() {
    return 'RcoluActivation';
  }

  constructor(config) {
    super(config);
    this.dim = config.dim ?? DEFAULT_GPT_CONFIG.coluDim;
    this.eps = config.eps ?? DEFAULT_GPT_CONFIG.coluEps;
  }

  call(inputs) {
    return tf.tidy(() => {
      const x = Array.isArray(inputs) ? inputs[0] : inputs;
      return rcoluActivationTensor(x, this.dim, this.eps);
    });
  }

  getConfig() {
    const baseConfig = super.getConfig();
    return Object.assign({}, baseConfig, {
      dim: this.dim,
      eps: this.eps,
    });
  }
}

tf.serialization.registerClass(RcoluActivation);

function rcoluActivationTensor(tensor, dim = 4, eps = 1e-6) {
  return tf.tidy(() => {
    const shape = tensor.shape;
    const C = shape[shape.length - 1];
    
    if (C % dim !== 0) {
      throw new Error(`RCoLU requires last dimension divisible by ${dim}, received ${C}`);
    }
    
    const G = C / dim;
    const newShape = shape.slice(0, -1).concat([G, dim]);
    
    // Reshape x to group structure: (..., G, dim)
    const xg = tensor.reshape(newShape);
    
    // Compute mu = mean(xg, axis=-1, keepdims=True)  # (..., G, 1)
    const mu = tf.mean(xg, -1, true);
    
    // Compute vn = mu * sqrt(dim)
    const sqrtDim = Math.sqrt(dim);
    const vn = mu.mul(sqrtDim);
    
    // Compute w = xg - mu
    const w = xg.sub(mu);
    
    // Compute wn = norm(w, axis=-1, keepdims=True)
    const wn = tf.norm(w, 'euclidean', -1, true);
    
    // Compute m = clamp(vn / (wn + eps), 0, 1)
    const m = tf.clipByValue(vn.div(wn.add(eps)), 0, 1);
    
    // Compute result: xg = mu + w * m
    const result = mu.add(w.mul(m));
    
    // Reshape back to original shape
    return result.reshape(shape);
  });
}

function coluActivationTensor(tensor, dim = 4, eps = 1e-7) {
  return tf.tidy(() => {
    const shape = tensor.shape;
    const lastDim = shape[shape.length - 1];
    if (lastDim % dim !== 0) {
      throw new Error(`CoLU requires last dimension divisible by ${dim}, received ${lastDim}`);
    }

    const numGroups = lastDim / dim;
    const splitSizes = [numGroups, lastDim - numGroups];
    const [yRaw, xRaw] = tf.split(tensor, splitSizes, -1);

    const xShape = xRaw.shape;
    const yShape = yRaw.shape;
    const reshapedXShape = xShape.slice(0, -1).concat([numGroups, dim - 1]);
    const reshapedYShape = yShape.slice(0, -1).concat([numGroups, 1]);

    const x = xRaw.reshape(reshapedXShape);
    const y = yRaw.reshape(reshapedYShape);

    const xn = tf.sum(tf.square(x), -1, true).sqrt().add(eps);
    const mask = y.div(xn).clipByValue(0, 1);
    const gated = mask.mul(x).reshape(xShape);

    return tf.concat([yRaw, gated], -1);
  });
}

export function buildModel(config) {
  const {
    vocabSize,
    blockSize = DEFAULT_GPT_CONFIG.blockSize,
    embeddingSize = DEFAULT_GPT_CONFIG.embeddingSize,
    numHeads = DEFAULT_GPT_CONFIG.numHeads,
    nLayers = DEFAULT_GPT_CONFIG.nLayers,
    ffHiddenSize = DEFAULT_GPT_CONFIG.ffHiddenSize,
    learningRate = DEFAULT_GPT_CONFIG.learningRate,
    activation = DEFAULT_GPT_CONFIG.activation,
    coluDim = DEFAULT_GPT_CONFIG.coluDim,
    coluEps = DEFAULT_GPT_CONFIG.coluEps,
  } = config;

  if (!vocabSize) {
    throw new Error('vocabSize must be provided to build GPT model');
  }
  const useCustomActivation = activation === 'colu' || activation === 'rcolu';

  const tokens = tf.input({shape: [blockSize], dtype: 'int32', name: 'tokens'});
  const positions = tf.input({shape: [blockSize], dtype: 'int32', name: 'positions'});

  const tokenEmbedding = tf.layers.embedding({
    inputDim: vocabSize,
    outputDim: embeddingSize,
    embeddingsInitializer: 'glorotUniform',
    name: 'token_embedding',
  }).apply(tokens);

  const positionEmbedding = tf.layers.embedding({
    inputDim: blockSize,
    outputDim: embeddingSize,
    embeddingsInitializer: 'glorotUniform',
    name: 'position_embedding',
  }).apply(positions);

  let x = tf.layers.add().apply([tokenEmbedding, positionEmbedding]);

  for (let layer = 0; layer < nLayers; layer++) {
    const attnLayer = new CausalSelfAttention({
      embedDim: embeddingSize,
      numHeads,
      blockSize,
      name: `causal_self_attention_${layer}`,
    });

    const attnOut = attnLayer.apply(x);
    const attnResidual = tf.layers.add().apply([x, attnOut]);
    const attnNorm = tf.layers.layerNormalization({
      axis: -1,
      epsilon: 1e-5,
      name: `attn_norm_${layer}`,
    }).apply(attnResidual);

    let ff1 = tf.layers.dense({
      units: ffHiddenSize,
      activation: useCustomActivation ? 'linear' : activation,
      name: `ffn_dense1_${layer}`,
    }).apply(attnNorm);
    
    if (activation === 'colu') {
      const coluLayer = new ColuActivation({dim: coluDim, eps: coluEps, name: `colu_activation_${layer}`});
      ff1 = coluLayer.apply(ff1);
    } else if (activation === 'rcolu') {
      const rcoluLayer = new RcoluActivation({dim: coluDim, eps: coluEps, name: `rcolu_activation_${layer}`});
      ff1 = rcoluLayer.apply(ff1);
    }
    const ff2 = tf.layers.dense({
      units: embeddingSize,
      name: `ffn_dense2_${layer}`,
    }).apply(ff1);

    const ffResidual = tf.layers.add().apply([attnNorm, ff2]);
    x = tf.layers.layerNormalization({
      axis: -1,
      epsilon: 1e-5,
      name: `ffn_norm_${layer}`,
    }).apply(ffResidual);
  }

  const logits = tf.layers.dense({
    units: vocabSize,
    activation: 'softmax',
    name: 'lm_head',
  }).apply(x);

  const model = tf.model({
    inputs: [tokens, positions],
    outputs: logits,
    name: `tiny_gpt_${config.activation || 'relu'}`,
  });

  const optimizer = tf.train.adam(learningRate);
  model.compile({
    optimizer,
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
  });

  return model;
}

export function createPositionTensor(batchSize, blockSize) {
  const data = new Int32Array(batchSize * blockSize);
  for (let i = 0; i < batchSize; i++) {
    const offset = i * blockSize;
    for (let j = 0; j < blockSize; j++) {
      data[offset + j] = j;
    }
  }
  return tf.tensor2d(data, [batchSize, blockSize], 'int32');
}

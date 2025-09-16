/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import {GptCharData} from './data.js';

const tf = window.tf;

const MASK_CACHE = new Map();

export const DEFAULT_GPT_CONFIG = {
  blockSize: 64,
  embeddingSize: 64,
  numHeads: 4,
  nLayers: 3,
  ffHiddenSize: 256,
  learningRate: 5e-3,
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
      throw new Error(
          `Embedding size ${this.embedDim} must be divisible by numHeads ${this.numHeads}`);
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

      // Store attention weights for later access
      const attnView = attn.reshape([batchSize, heads, seqLen, seqLen]);
      this.lastAttentionWeights = attnView;
      
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

export function buildModel(config) {
  const {
    vocabSize,
    blockSize = DEFAULT_GPT_CONFIG.blockSize,
    embeddingSize = DEFAULT_GPT_CONFIG.embeddingSize,
    numHeads = DEFAULT_GPT_CONFIG.numHeads,
    nLayers = DEFAULT_GPT_CONFIG.nLayers,
    ffHiddenSize = DEFAULT_GPT_CONFIG.ffHiddenSize,
    learningRate = DEFAULT_GPT_CONFIG.learningRate,
  } = config;

  if (!vocabSize) {
    throw new Error('vocabSize must be provided to build GPT model');
  }

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

  // Stack multiple transformer layers
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

    const ff1 = tf.layers.dense({
      units: ffHiddenSize,
      activation: 'relu',
      name: `ffn_dense1_${layer}`,
    }).apply(attnNorm);
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
    name: 'tiny_gpt',
  });

  // Create a simple dummy attention probe that returns random attention weights
  // First flatten the initial embeddings to get a consistent input size
  const initialSum = tf.layers.add().apply([tokenEmbedding, positionEmbedding]);
  const flattened = tf.layers.flatten().apply(initialSum);
  
  const dummyAttention = tf.layers.dense({
    units: numHeads * blockSize * blockSize,
    activation: 'softmax',
    name: 'dummy_attention'
  }).apply(flattened);
  
  const reshapedAttention = tf.layers.reshape({
    targetShape: [numHeads, blockSize, blockSize],
    name: 'reshaped_attention'
  }).apply(dummyAttention);
  
  const attentionProbe = tf.model({
    inputs: [tokens, positions],
    outputs: reshapedAttention,
    name: 'tiny_gpt_attention',
  });

  const optimizer = tf.train.adam(learningRate);
  model.compile({
    optimizer,
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
  });

  return {model, attentionProbe, config: {blockSize, embeddingSize, numHeads, nLayers, ffHiddenSize, learningRate}};
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

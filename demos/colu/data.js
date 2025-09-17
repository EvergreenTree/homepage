const tf = window.tf;

const DEFAULT_BLOCK_SIZE = 64;
const TRAIN_FRACTION = 0.9;

export class GptCharData {
  constructor(opts = {}) {
    this.blockSize = opts.blockSize || DEFAULT_BLOCK_SIZE;
    this.trainFraction = opts.trainFraction || TRAIN_FRACTION;

    this.text = '';
    this.vocab = [];
    this.stoi = new Map();
    this.itos = [];
    this.tokens = null;
    this.trainTokens = null;
    this.valTokens = null;
  }

  async load() {
    const response = await fetch('./input.txt');
    if (!response.ok) {
      throw new Error(`Failed to load input.txt: ${response.status}`);
    }
    this.text = await response.text();
    this.#buildVocab();
    this.#tokenise();
    this.#splitTrainVal();
    return this;
  }

  #buildVocab() {
    this.vocab = [];
    this.stoi.clear();
    this.itos = [];

    for (let i = 0; i < this.text.length; i++) {
      const ch = this.text[i];
      if (!this.stoi.has(ch)) {
        const id = this.vocab.length;
        this.vocab.push(ch);
        this.stoi.set(ch, id);
        this.itos[id] = ch;
      }
    }
    this.vocabSize = this.vocab.length;
  }

  #tokenise() {
    const arr = new Int32Array(this.text.length);
    for (let i = 0; i < this.text.length; i++) {
      const ch = this.text[i];
      const id = this.stoi.get(ch);
      if (id === undefined) {
        throw new Error(`Encountered unknown character during tokenisation: "${ch}"`);
      }
      arr[i] = id;
    }
    this.tokens = arr;
  }

  #splitTrainVal() {
    const split = Math.floor(this.tokens.length * this.trainFraction);
    this.trainTokens = this.tokens.subarray(0, split);
    this.valTokens = this.tokens.subarray(split);
  }

  sampleBatch(split = 'train', batchSize = 64, blockSize = this.blockSize) {
    const src = split === 'train' ? this.trainTokens : this.valTokens;
    if (!src || src.length < blockSize + 1) {
      throw new Error('Not enough data to sample the requested batch.');
    }
    const xsArr = new Int32Array(batchSize * blockSize);
    const ysArr = new Int32Array(batchSize * blockSize);
    const maxIndex = src.length - blockSize - 1;

    for (let b = 0; b < batchSize; b++) {
      const start = Math.floor(Math.random() * maxIndex);
      const x = src.subarray(start, start + blockSize);
      const y = src.subarray(start + 1, start + blockSize + 1);
      xsArr.set(x, b * blockSize);
      ysArr.set(y, b * blockSize);
    }

    const xs = tf.tensor2d(xsArr, [batchSize, blockSize], 'int32');
    const ys = tf.tensor2d(ysArr, [batchSize, blockSize], 'int32');
    return {xs, ys};
  }

  nextTrainBatch(batchSize = 64, blockSize = this.blockSize) {
    return this.sampleBatch('train', batchSize, blockSize);
  }

  nextValBatch(batchSize = 64, blockSize = this.blockSize) {
    return this.sampleBatch('val', batchSize, blockSize);
  }

  stats() {
    return {
      vocabSize: this.vocab.length,
      blockSize: this.blockSize,
      textLength: this.text.length,
      tokens: this.tokens?.length || 0,
      trainTokens: this.trainTokens?.length || 0,
      valTokens: this.valTokens?.length || 0,
      trainFraction: this.trainFraction,
    };
  }
}

export async function loadData(opts) {
  const data = new GptCharData(opts);
  await data.load();
  return data;
}

import * as fs from 'fs';
import * as path from 'path';
import { Sample } from './datasets';

const MNIST_DIR = path.join(__dirname, '..', 'data', 'mnist');

export function loadMnist(maxTrain = 1000, maxTest = 200): { train: Sample[]; test: Sample[] } {
  if (!fs.existsSync(MNIST_DIR)) {
    console.error('MNIST data not found. Run: npm run download-mnist');
    process.exit(1);
  }

  const trainImages = readImages(path.join(MNIST_DIR, 'train-images-idx3-ubyte'));
  const trainLabels = readLabels(path.join(MNIST_DIR, 'train-labels-idx1-ubyte'));
  const testImages = readImages(path.join(MNIST_DIR, 't10k-images-idx3-ubyte'));
  const testLabels = readLabels(path.join(MNIST_DIR, 't10k-labels-idx1-ubyte'));

  const train = stratifiedSample(trainImages, trainLabels, maxTrain);
  const test = stratifiedSample(testImages, testLabels, maxTest);

  console.log(`MNIST loaded: ${train.length} train, ${test.length} test (28x28 = 784 inputs, 10 classes)`);
  return { train, test };
}

function stratifiedSample(images: number[][], labels: number[], max: number): Sample[] {
  const perClass = Math.ceil(max / 10);
  const buckets: Map<number, number[][]> = new Map();

  for (let i = 0; i < labels.length; i++) {
    const label = labels[i];
    if (!buckets.has(label)) buckets.set(label, []);
    const bucket = buckets.get(label)!;
    if (bucket.length < perClass) {
      bucket.push(images[i]);
    }
  }

  const samples: Sample[] = [];
  for (const [label, imgs] of buckets) {
    for (const img of imgs) {
      samples.push({ inputs: img, target: oneHot(label, 10) });
    }
  }

  // shuffle so training isn't class-ordered
  for (let i = samples.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [samples[i], samples[j]] = [samples[j], samples[i]];
  }

  return samples;
}

function readImages(filePath: string): number[][] {
  const buf = fs.readFileSync(filePath);
  const magic = buf.readUInt32BE(0);
  if (magic !== 2051) throw new Error(`Invalid image file (magic: ${magic})`);
  const count = buf.readUInt32BE(4);
  const rows = buf.readUInt32BE(8);
  const cols = buf.readUInt32BE(12);
  const size = rows * cols;

  const images: number[][] = [];
  for (let i = 0; i < count; i++) {
    const offset = 16 + i * size;
    const pixels: number[] = [];
    for (let j = 0; j < size; j++) {
      pixels.push(buf[offset + j] / 255);
    }
    images.push(pixels);
  }
  return images;
}

function readLabels(filePath: string): number[] {
  const buf = fs.readFileSync(filePath);
  const magic = buf.readUInt32BE(0);
  if (magic !== 2049) throw new Error(`Invalid label file (magic: ${magic})`);
  const count = buf.readUInt32BE(4);

  const labels: number[] = [];
  for (let i = 0; i < count; i++) {
    labels.push(buf[8 + i]);
  }
  return labels;
}

function oneHot(label: number, classes: number): number[] {
  const vec = new Array(classes).fill(0);
  vec[label] = 1;
  return vec;
}

export function evaluate(network: { forward(inputs: number[]): number[] }, samples: Sample[], showCount = 0): void {
  let correct = 0;
  const perClass = new Array(10).fill(0);
  const perClassTotal = new Array(10).fill(0);
  const errors: { sample: Sample; predicted: number; confidence: number }[] = [];

  for (const sample of samples) {
    const result = network.forward(sample.inputs);
    const predicted = result.indexOf(Math.max(...result));
    const actual = sample.target.indexOf(1);
    perClassTotal[actual]++;
    if (predicted === actual) {
      correct++;
      perClass[actual]++;
    } else {
      errors.push({ sample, predicted, confidence: result[predicted] });
    }
  }

  console.log(`\nTest accuracy: ${correct}/${samples.length} (${(100 * correct / samples.length).toFixed(1)}%)\n`);
  console.log('Per digit:');
  for (let d = 0; d < 10; d++) {
    const pct = perClassTotal[d] > 0 ? (100 * perClass[d] / perClassTotal[d]).toFixed(0) : '-';
    const bar = '█'.repeat(Math.round(perClass[d] / Math.max(1, perClassTotal[d]) * 20));
    console.log(`  ${d}: ${pct.padStart(3)}%  ${bar}  (${perClass[d]}/${perClassTotal[d]})`);
  }

  if (showCount <= 0) return;

  // show some correct predictions
  const shown = new Set<number>();
  console.log('\n--- Correct predictions (never seen during training) ---');
  for (const sample of samples) {
    const result = network.forward(sample.inputs);
    const predicted = result.indexOf(Math.max(...result));
    const actual = sample.target.indexOf(1);
    if (predicted === actual && !shown.has(actual)) {
      shown.add(actual);
      renderDigit(sample.inputs, actual, predicted, result[predicted]);
      if (shown.size >= Math.min(showCount, 10)) break;
    }
  }

  // show some errors
  if (errors.length > 0) {
    console.log('\n--- Errors ---');
    for (const err of errors.slice(0, Math.min(3, showCount))) {
      const actual = err.sample.target.indexOf(1);
      renderDigit(err.sample.inputs, actual, err.predicted, err.confidence);
    }
  }
}

const SHADES = ' .:-=+*#%@';

function renderDigit(pixels: number[], actual: number, predicted: number, confidence: number): void {
  const ok = actual === predicted;
  const label = ok
    ? `  Actual: ${actual}  Predicted: ${predicted} ✓  (${(confidence * 100).toFixed(0)}%)`
    : `  Actual: ${actual}  Predicted: ${predicted} ✗  (${(confidence * 100).toFixed(0)}%)`;
  console.log(label);

  for (let row = 0; row < 28; row++) {
    let line = '  ';
    for (let col = 0; col < 28; col++) {
      const v = pixels[row * 28 + col];
      line += SHADES[Math.min(Math.floor(v * SHADES.length), SHADES.length - 1)];
    }
    console.log(line);
  }
  console.log('');
}

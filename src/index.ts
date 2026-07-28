import * as readline from 'readline';
import { Network } from './network';
import { Dataset, DATASETS, Sample } from './datasets';
import { loadMnist, evaluate } from './mnist';

// --- Arg parsing ---

const args = process.argv.slice(2);
const flags: Record<string, string | boolean> = {};
for (let i = 0; i < args.length; i++) {
  if (args[i].startsWith('--')) {
    const key = args[i].slice(2);
    const next = args[i + 1];
    if (next && !next.startsWith('--')) {
      flags[key] = next;
      i++;
    } else {
      flags[key] = true;
    }
  } else if (!flags.dataset) {
    flags.dataset = args[i];
  }
}

if (flags.help) {
  console.log(`Usage: node dist/index.js [dataset] [options]

Datasets:${Object.entries(DATASETS).map(([k, d]) => `\n  ${k.padEnd(12)} ${d.description}`).join('')}

Options:
  --lr <n>            Learning rate (default: 0.5)
  --iterations <n>    Max iterations (default: 50000)
  --hidden <n>        Hidden layer size (default: 8)
  --layers <n,n,...>  Custom architecture (e.g. 8,5,3,5,8)
  --samples <n>       Training samples for MNIST (default: 1000)
  --show <n>          Show n test predictions as ASCII art (MNIST)
  --inspect           Show bottleneck activations after training
  --debug             Show full network state instead of progress bar
  --help              Show this help`);
  process.exit(0);
}

// --- Config ---

const datasetKey = (flags.dataset as string) || 'xor';
if (!DATASETS[datasetKey]) {
  console.error(`Unknown dataset "${datasetKey}". Available: ${Object.keys(DATASETS).join(', ')}`);
  process.exit(1);
}

const learningRate = parseFloat(flags.lr as string) || 0.5;
const hiddenSize = parseInt(flags.hidden as string, 10) || 8;
const debug = !!flags.debug;
const inspect = !!flags.inspect;

// --- Load dataset ---

let samples: Sample[];
let testSamples: Sample[] | undefined;
let description: string;
let recommendedLayers: number[] | undefined;

if (datasetKey === 'mnist') {
  const sampleCount = parseInt(flags.samples as string, 10) || 1000;
  const mnist = loadMnist(sampleCount, Math.ceil(sampleCount / 5));
  samples = mnist.train;
  testSamples = mnist.test;
  description = DATASETS[datasetKey].description;
  recommendedLayers = [784, 128, 10];
} else {
  const ds = DATASETS[datasetKey] as Dataset;
  samples = ds.samples;
  description = ds.description;
  recommendedLayers = ds.recommendedLayers;
}

const maxIterations = parseInt(flags.iterations as string, 10) || (datasetKey === 'mnist' ? 100000 : 50000);

const inputSize = samples[0].inputs.length;
const outputSize = samples[0].target.length;

let layers: number[];
if (flags.layers) {
  layers = (flags.layers as string).split(',').map(Number);
  if (layers[0] !== inputSize) {
    console.error(`First layer (${layers[0]}) must match input size (${inputSize})`);
    process.exit(1);
  }
  if (layers[layers.length - 1] !== outputSize) {
    console.error(`Last layer (${layers[layers.length - 1]}) must match output size (${outputSize})`);
    process.exit(1);
  }
} else if (recommendedLayers) {
  layers = recommendedLayers;
} else {
  layers = [inputSize, hiddenSize, outputSize];
}

console.log(`Dataset: ${datasetKey} - ${description}`);
console.log(`Network: ${layers.join(' -> ')}  |  LR: ${learningRate}  |  Iterations: ${maxIterations}\n`);

// --- Display ---

function bgColor(r: number, g: number, b: number, text: string): string {
  return `\x1b[48;2;${r};${g};${b}m${text}\x1b[0m`;
}

const precisionHistory: number[] = [];
const REFRESH_EVERY = 97;

function displayProgress(target: number[], result: number[], iteration: number): void {
  const avg = precisionHistory.reduce((a, b) => a + b) / precisionHistory.length;

  const barWidth = 40;
  const filled = Math.round((avg / 100) * barWidth);
  const bar = '█'.repeat(filled) + '░'.repeat(barWidth - filled);

  const predicted = result.indexOf(Math.max(...result));
  const actual = target.indexOf(1);
  const classInfo = target.length > 2 ? `  ${actual}->${predicted}` : '';

  const targetStr = target.length <= 8
    ? `T:[${target.map(t => t.toFixed(0)).join(' ')}]`
    : `T:${actual}`;
  const resultStr = target.length <= 8
    ? `R:[${result.map(r => r.toFixed(3)).join(' ')}]`
    : `P:${predicted} (${(result[predicted] * 100).toFixed(0)}%)`;

  const epochInfo = samples.length > 10
    ? `[${Math.floor(iteration / samples.length) + 1}e] `
    : '';

  const line = ` ${epochInfo}${avg.toFixed(1)}% ${bar}  ${targetStr}  ${resultStr}${classInfo} `;

  const r = Math.round(200 - 150 * (avg / 100));
  const g = Math.round(130 * (avg / 100));

  readline.cursorTo(process.stdout, 0);
  readline.clearLine(process.stdout, 0);
  process.stdout.write(bgColor(r, g, 0, line));
}

// --- Training ---

const network = new Network(layers);

for (let i = 0; i < maxIterations; i++) {
  const sample = samples[i % samples.length];
  const result = network.forward(sample.inputs);

  let precision = 0;
  for (let j = 0; j < sample.target.length; j++) {
    precision += 1 - Math.abs(sample.target[j] - result[j]);
  }
  precision /= sample.target.length;

  precisionHistory.push(precision * 100);
  if (precisionHistory.length > 20) precisionHistory.shift();

  if (debug) {
    console.log(network.toString());
  } else if (i % REFRESH_EVERY === 0) {
    // Redrawing costs more than training does, and a stride that shares no
    // factor with the sample count keeps the displayed sample moving.
    displayProgress(sample.target, result, i);
  }

  network.backpropagate(sample.target, learningRate);
}

console.log('');

// --- Evaluate on test set ---

if (testSamples && testSamples.length > 0) {
  const showCount = parseInt(flags.show as string, 10) || 0;
  evaluate(network, testSamples, showCount);
}

// --- Inspect bottleneck ---

if (inspect) {
  const bottleneckIndex = Math.floor(network.layers.length / 2);
  console.log(`\nBottleneck (layer ${bottleneckIndex}, ${network.bottleneckLayer.size} neurons):\n`);

  const inspectSamples = samples.length > 20 ? samples.slice(0, 20) : samples;
  for (const sample of inspectSamples) {
    network.forward(sample.inputs);
    const bottleneck = network.bottleneckLayer.activations;
    const rounded = bottleneck.map(v => Math.round(v));
    const output = network.outputLayer.activations;

    const inputStr = sample.inputs.length > 20
      ? `digit ${sample.target.indexOf(1)}`
      : `[${sample.inputs.join('')}]`;

    console.log(
      `  Input: ${inputStr}` +
      `  Bottleneck: [${bottleneck.map(v => v.toFixed(3)).join(', ')}]` +
      `  ~ [${rounded.join('')}]` +
      `  Output: [${output.map(v => v.toFixed(3)).join(', ')}]`,
    );
  }
}

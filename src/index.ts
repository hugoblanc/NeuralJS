import * as readline from 'readline';
import { Network } from './network';
import { DATASETS } from './datasets';

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
  --inspect           Show bottleneck activations after training
  --debug             Show full network state instead of progress bar
  --help              Show this help`);
  process.exit(0);
}

// --- Config ---

const datasetKey = (flags.dataset as string) || 'xor';
const dataset = DATASETS[datasetKey];
if (!dataset) {
  console.error(`Unknown dataset "${datasetKey}". Available: ${Object.keys(DATASETS).join(', ')}`);
  process.exit(1);
}

const learningRate = parseFloat(flags.lr as string) || 0.5;
const maxIterations = parseInt(flags.iterations as string, 10) || 50000;
const hiddenSize = parseInt(flags.hidden as string, 10) || 8;
const debug = !!flags.debug;
const inspect = !!flags.inspect;

const inputSize = dataset.samples[0].inputs.length;
const outputSize = dataset.samples[0].target.length;

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
} else if (dataset.recommendedLayers) {
  layers = dataset.recommendedLayers;
} else {
  layers = [inputSize, hiddenSize, outputSize];
}

console.log(`Dataset: ${datasetKey} - ${dataset.description}`);
console.log(`Network: ${layers.join(' -> ')}  |  LR: ${learningRate}  |  Iterations: ${maxIterations}\n`);

// --- Display ---

function bgColor(r: number, g: number, b: number, text: string): string {
  return `\x1b[48;2;${r};${g};${b}m${text}\x1b[0m`;
}

const precisionHistory: number[] = [];

function displayProgress(precision: number, target: number[], result: number[]): void {
  precisionHistory.push(precision * 100);
  if (precisionHistory.length > 10) precisionHistory.shift();
  const avg = precisionHistory.reduce((a, b) => a + b) / precisionHistory.length;

  const barWidth = 40;
  const filled = Math.round((avg / 100) * barWidth);
  const bar = '█'.repeat(filled) + '░'.repeat(barWidth - filled);

  const targetStr = target.map(t => t.toFixed(0)).join(' ');
  const resultStr = result.map(r => r.toFixed(3)).join(' ');
  const line = ` ${avg.toFixed(1)}% ${bar}  T:[${targetStr}]  R:[${resultStr}] `;

  const r = Math.round(200 - 150 * (avg / 100));
  const g = Math.round(130 * (avg / 100));

  readline.cursorTo(process.stdout, 0);
  readline.clearLine(process.stdout, 0);
  process.stdout.write(bgColor(r, g, 0, line));
}

// --- Training ---

const network = new Network(layers);

for (let i = 0; i < maxIterations; i++) {
  const sample = dataset.samples[i % dataset.samples.length];
  const result = network.forward(sample.inputs);

  let precision = 0;
  for (let j = 0; j < sample.target.length; j++) {
    precision += 1 - Math.abs(sample.target[j] - result[j]);
  }
  precision /= sample.target.length;

  if (debug) {
    console.log(network.toString());
  } else {
    displayProgress(precision, sample.target, result);
  }

  network.backpropagate(sample.target, learningRate);
}

console.log('');

// --- Inspect bottleneck ---

if (inspect) {
  const bottleneckIndex = Math.floor(network.layers.length / 2);
  const bottleneckSize = network.layers[bottleneckIndex].neurons.length;
  console.log(`\nBottleneck (layer ${bottleneckIndex}, ${bottleneckSize} neurons):\n`);

  for (const sample of dataset.samples) {
    network.forward(sample.inputs);
    const bottleneck = network.layers[bottleneckIndex].neurons.map(n => n.activation);
    const rounded = bottleneck.map(v => Math.round(v));
    const output = network.outputLayer.neurons.map(n => n.activation);

    console.log(
      `  Input: [${sample.inputs.join('')}]` +
      `  Bottleneck: [${bottleneck.map(v => v.toFixed(3)).join(', ')}]` +
      `  ~ [${rounded.join('')}]` +
      `  Output: [${output.map(v => v.toFixed(3)).join(', ')}]`,
    );
  }
}

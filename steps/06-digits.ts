// Step 06 - Recognizing Digits
// =============================
// Everything we've built so far was on toy problems.
// Now we apply it to the same problem 3Blue1Brown uses:
// recognizing handwritten digits from the MNIST dataset.
//
// 784 pixels (28x28) -> 128 hidden neurons -> 10 digit classes
//
// Before running this step, download MNIST:
//   npm run download-mnist
//
// 3Blue1Brown reference: all 3 chapters use this as the running example
// https://www.youtube.com/watch?v=aircAruvnKk

import * as fs from "fs";
import * as path from "path";
import * as readline from "readline";

// --- Same network code as before (no imports, everything is here) ---

function sigmoid(x: number): number { return 1 / (1 + Math.exp(-x)); }
function sigmoidDeriv(x: number): number { const s = sigmoid(x); return s * (1 - s); }

interface Synapse { weight: number; parent: Neuron; }
class Neuron {
  bias = 1; sum = 0; activation = 0; delta = 0;
  synapses: Synapse[] = [];
  constructor(parents?: Neuron[]) {
    if (!parents) return;
    this.synapses = parents.map(p => ({ weight: Math.random() * 2 - 1, parent: p }));
  }
}
class Layer {
  neurons: Neuron[];
  constructor(size: number, parent?: Layer) {
    this.neurons = Array.from({ length: size }, () => new Neuron(parent?.neurons));
  }
}
class Network {
  layers: Layer[];
  constructor(sizes: number[]) {
    this.layers = []; let prev: Layer | undefined;
    for (const s of sizes) { const l = new Layer(s, prev); this.layers.push(l); prev = l; }
  }
  forward(inputs: number[]): number[] {
    this.layers[0].neurons.forEach((n, i) => { n.activation = inputs[i]; });
    for (let l = 1; l < this.layers.length; l++) {
      for (const n of this.layers[l].neurons) {
        let s = 0;
        for (const syn of n.synapses) s += syn.parent.activation * syn.weight;
        n.sum = s + n.bias; n.activation = sigmoid(n.sum);
      }
    }
    return this.layers[this.layers.length - 1].neurons.map(n => n.activation);
  }
  backpropagate(target: number[], lr: number): void {
    const out = this.layers[this.layers.length - 1].neurons;
    for (let i = 0; i < out.length; i++)
      out[i].delta = 2 * (out[i].activation - target[i]) * sigmoidDeriv(out[i].sum);
    for (let l = this.layers.length - 2; l >= 1; l--)
      for (let j = 0; j < this.layers[l].neurons.length; j++) {
        const n = this.layers[l].neurons[j]; let err = 0;
        for (const nn of this.layers[l + 1].neurons) err += nn.synapses[j].weight * nn.delta;
        n.delta = err * sigmoidDeriv(n.sum);
      }
    for (let l = 1; l < this.layers.length; l++)
      for (const n of this.layers[l].neurons) {
        n.bias -= n.delta * lr;
        for (const syn of n.synapses) syn.weight -= n.delta * syn.parent.activation * lr;
      }
  }
}

// --- MNIST loader (reads the raw binary files) ---

interface Sample { inputs: number[]; target: number[]; }

function readImages(filePath: string): number[][] {
  const buf = fs.readFileSync(filePath);
  const count = buf.readUInt32BE(4);
  const rows = buf.readUInt32BE(8);
  const cols = buf.readUInt32BE(12);
  const size = rows * cols;
  const images: number[][] = [];
  for (let i = 0; i < count; i++) {
    const offset = 16 + i * size;
    const pixels: number[] = [];
    for (let j = 0; j < size; j++) pixels.push(buf[offset + j] / 255);
    images.push(pixels);
  }
  return images;
}

function readLabels(filePath: string): number[] {
  const buf = fs.readFileSync(filePath);
  const count = buf.readUInt32BE(4);
  const labels: number[] = [];
  for (let i = 0; i < count; i++) labels.push(buf[8 + i]);
  return labels;
}

function oneHot(label: number): number[] {
  const v = new Array(10).fill(0);
  v[label] = 1;
  return v;
}

function loadMnist(dir: string, maxTrain: number, maxTest: number) {
  const trainImages = readImages(path.join(dir, "train-images-idx3-ubyte"));
  const trainLabels = readLabels(path.join(dir, "train-labels-idx1-ubyte"));
  const testImages = readImages(path.join(dir, "t10k-images-idx3-ubyte"));
  const testLabels = readLabels(path.join(dir, "t10k-labels-idx1-ubyte"));

  const train: Sample[] = [];
  for (let i = 0; i < Math.min(maxTrain, trainImages.length); i++) {
    train.push({ inputs: trainImages[i], target: oneHot(trainLabels[i]) });
  }
  const test: Sample[] = [];
  for (let i = 0; i < Math.min(maxTest, testImages.length); i++) {
    test.push({ inputs: testImages[i], target: oneHot(testLabels[i]) });
  }
  return { train, test };
}

// --- Go ---

const MNIST_DIR = path.join(__dirname, "..", "data", "mnist");
if (!fs.existsSync(MNIST_DIR)) {
  console.error("MNIST data not found. Run: npm run download-mnist");
  process.exit(1);
}

const trainCount = parseInt(process.argv[2] || "1000", 10);
const iterations = parseInt(process.argv[3] || "50000", 10);
const lr = parseFloat(process.argv[4] || "0.5");

const { train, test } = loadMnist(MNIST_DIR, trainCount, 200);
console.log(`=== MNIST: ${train.length} train, ${test.length} test ===`);
console.log(`Network: 784 -> 128 -> 10 | LR: ${lr} | Iterations: ${iterations}\n`);

const net = new Network([784, 128, 10]);
const history: number[] = [];

function bg(r: number, g: number, b: number, text: string): string {
  return `\x1b[48;2;${r};${g};${b}m${text}\x1b[0m`;
}

for (let i = 0; i < iterations; i++) {
  const sample = train[i % train.length];
  const result = net.forward(sample.inputs);
  const predicted = result.indexOf(Math.max(...result));
  const actual = sample.target.indexOf(1);

  let precision = 0;
  for (let j = 0; j < sample.target.length; j++)
    precision += 1 - Math.abs(sample.target[j] - result[j]);
  precision /= sample.target.length;

  history.push(precision * 100);
  if (history.length > 10) history.shift();
  const avg = history.reduce((a, b) => a + b) / history.length;

  const epoch = Math.floor(i / train.length) + 1;
  const barW = 40;
  const filled = Math.round((avg / 100) * barW);
  const bar = "█".repeat(filled) + "░".repeat(barW - filled);
  const line = ` [${epoch}e] ${avg.toFixed(1)}% ${bar}  actual:${actual} pred:${predicted} `;
  const red = Math.round(200 - 150 * (avg / 100));
  const green = Math.round(130 * (avg / 100));

  readline.cursorTo(process.stdout, 0);
  readline.clearLine(process.stdout, 0);
  process.stdout.write(bg(red, green, 0, line));

  net.backpropagate(sample.target, lr);
}

// --- Evaluate on test set ---

console.log("\n\n=== Test set evaluation ===\n");

let correct = 0;
const perClass = new Array(10).fill(0);
const perClassTotal = new Array(10).fill(0);

for (const sample of test) {
  const result = net.forward(sample.inputs);
  const predicted = result.indexOf(Math.max(...result));
  const actual = sample.target.indexOf(1);
  perClassTotal[actual]++;
  if (predicted === actual) { correct++; perClass[actual]++; }
}

console.log(`Accuracy: ${correct}/${test.length} (${(100 * correct / test.length).toFixed(1)}%)\n`);
for (let d = 0; d < 10; d++) {
  const pct = perClassTotal[d] > 0 ? (100 * perClass[d] / perClassTotal[d]).toFixed(0) : "-";
  const bar = "█".repeat(Math.round(perClass[d] / Math.max(1, perClassTotal[d]) * 20));
  console.log(`  ${d}: ${pct.padStart(3)}%  ${bar}  (${perClass[d]}/${perClassTotal[d]})`);
}

// --- Show a few predictions as ASCII art ---

console.log("\n=== Sample predictions (never seen during training) ===\n");
const SHADES = " .:-=+*#%@";
let shown = 0;
for (const sample of test) {
  if (shown >= 5) break;
  const result = net.forward(sample.inputs);
  const predicted = result.indexOf(Math.max(...result));
  const actual = sample.target.indexOf(1);
  const ok = predicted === actual ? "ok" : "MISS";
  console.log(`  Actual: ${actual}  Predicted: ${predicted}  ${ok}  (${(result[predicted] * 100).toFixed(0)}%)`);
  for (let row = 0; row < 28; row++) {
    let line = "  ";
    for (let col = 0; col < 28; col++) {
      const v = sample.inputs[row * 28 + col];
      line += SHADES[Math.min(Math.floor(v * SHADES.length), SHADES.length - 1)];
    }
    console.log(line);
  }
  console.log("");
  shown++;
}

console.log("--- Try it ---");
console.log("npx tsx steps/06-digits.ts 500 30000      (fast, ~80% accuracy)");
console.log("npx tsx steps/06-digits.ts 5000 200000     (slow, ~90% accuracy)");

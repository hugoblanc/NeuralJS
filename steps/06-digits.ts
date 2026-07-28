// Step 06 - Recognizing Digits
// ============================
// Everything so far ran on toy problems with at most sixteen samples.
// Now the real one: MNIST, 70000 handwritten digits, the same dataset
// 3Blue1Brown uses throughout his series.
//
//   784 pixels (28x28)  ->  128 hidden neurons  ->  10 digit classes
//
// Here is the thing worth noticing in this file: Synapse, Neuron, Layer
// and Network are copied from step 05 unchanged. Not simplified, not
// specialised - identical. The XOR network of step 04 had 2 inputs and
// 33 parameters; this one has 784 inputs and 101770. That needs no new
// maths at all - only new objects around the data:
//
//   Digit        - one labelled image, that knows how to draw itself
//   MnistLoader  - turns the raw IDX binary files into Digit objects
//   Classifier   - reads the 10 outputs as a decision plus a confidence
//   Evaluation   - the scoreboard, per digit
//
// Plus one new ProgressMetric, because "how close are the numbers" stops
// being a useful question once there are ten of them.
//
// Before running this step:  npm run download-mnist
//
// 3Blue1Brown reference: all three chapters use this as the running example
// https://www.youtube.com/watch?v=aircAruvnKk

import * as fs from "fs";
import * as path from "path";
import * as readline from "readline";

function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

function sigmoidDerivative(x: number): number {
  const s = sigmoid(x);
  return s * (1 - s);
}

/** Index of the largest value - which output neuron shouted loudest. */
function argmax(values: readonly number[]): number {
  let best = 0;
  for (let i = 1; i < values.length; i++) {
    if (values[i] > values[best]) best = i;
  }
  return best;
}

// --- The network, unchanged from step 05 ---

class Synapse {
  weight: number;

  constructor(
    readonly source: Neuron,
    readonly target: Neuron,
    weight: number = Math.random() * 2 - 1,
  ) {
    this.weight = weight;
  }

  get contribution(): number {
    return this.source.activation * this.weight;
  }

  get blame(): number {
    return this.weight * this.target.delta;
  }
}

class Neuron {
  bias = 1;
  sum = 0;
  activation = 0;
  delta = 0;
  readonly inputs: Synapse[] = [];
  readonly outputs: Synapse[] = [];

  connectFrom(source: Neuron, weight?: number): Synapse {
    const synapse = new Synapse(source, this, weight);
    this.inputs.push(synapse);
    source.outputs.push(synapse);
    return synapse;
  }

  receive(value: number): void {
    this.sum = value;
    this.activation = value;
  }

  activate(): number {
    let total = 0;
    for (const synapse of this.inputs) total += synapse.contribution;
    this.sum = total + this.bias;
    this.activation = sigmoid(this.sum);
    return this.activation;
  }

  computeOutputDelta(costGradient: number): void {
    this.delta = costGradient * sigmoidDerivative(this.sum);
  }

  computeHiddenDelta(): void {
    let error = 0;
    for (const synapse of this.outputs) error += synapse.blame;
    this.delta = error * sigmoidDerivative(this.sum);
  }

  applyGradient(learningRate: number): void {
    this.bias -= this.delta * learningRate;
    for (const synapse of this.inputs) {
      synapse.weight -= this.delta * synapse.source.activation * learningRate;
    }
  }
}

class Layer {
  readonly neurons: Neuron[];

  constructor(size: number, previous?: Layer) {
    this.neurons = Array.from({ length: size }, () => new Neuron());
    if (previous) this.fullyConnect(previous);
  }

  private fullyConnect(previous: Layer): void {
    for (const neuron of this.neurons) {
      for (const source of previous.neurons) {
        neuron.connectFrom(source);
      }
    }
  }

  get size(): number {
    return this.neurons.length;
  }

  get activations(): number[] {
    return this.neurons.map(neuron => neuron.activation);
  }

  receive(values: number[]): void {
    this.neurons.forEach((neuron, i) => neuron.receive(values[i]));
  }

  activate(): void {
    for (const neuron of this.neurons) neuron.activate();
  }

  computeOutputDeltas(costGradient: number[]): void {
    this.neurons.forEach((neuron, i) => neuron.computeOutputDelta(costGradient[i]));
  }

  computeHiddenDeltas(): void {
    for (const neuron of this.neurons) neuron.computeHiddenDelta();
  }

  applyGradient(learningRate: number): void {
    for (const neuron of this.neurons) neuron.applyGradient(learningRate);
  }
}

interface Sample {
  readonly inputs: number[];
  readonly target: number[];
}

interface CostFunction {
  readonly name: string;
  evaluate(output: number[], target: number[]): number;
  gradient(output: number[], target: number[]): number[];
}

class MeanSquaredError implements CostFunction {
  readonly name = "mean squared error";

  evaluate(output: number[], target: number[]): number {
    let total = 0;
    for (let i = 0; i < target.length; i++) {
      const error = output[i] - target[i];
      total += error * error;
    }
    return total / target.length;
  }

  gradient(output: number[], target: number[]): number[] {
    return target.map((t, i) => 2 * (output[i] - t));
  }
}

class Network {
  readonly layers: Layer[];

  constructor(
    sizes: number[],
    readonly cost: CostFunction = new MeanSquaredError(),
  ) {
    this.layers = [];
    let previous: Layer | undefined;
    for (const size of sizes) {
      previous = new Layer(size, previous);
      this.layers.push(previous);
    }
  }

  get inputLayer(): Layer {
    return this.layers[0];
  }

  get outputLayer(): Layer {
    return this.layers[this.layers.length - 1];
  }

  get computedLayers(): Layer[] {
    return this.layers.slice(1);
  }

  get weightCount(): number {
    return this.layers.reduce(
      (total, layer) => total + layer.neurons.reduce((n, neuron) => n + neuron.inputs.length, 0),
      0,
    );
  }

  /** One bias per neuron, except in the input layer where nothing is computed. */
  get biasCount(): number {
    return this.computedLayers.reduce((total, layer) => total + layer.size, 0);
  }

  /** Everything training is allowed to change. */
  get parameterCount(): number {
    return this.weightCount + this.biasCount;
  }

  forward(inputs: number[]): number[] {
    this.inputLayer.receive(inputs);
    for (const layer of this.computedLayers) layer.activate();
    return this.outputLayer.activations;
  }

  backpropagate(target: number[], learningRate: number): void {
    this.outputLayer.computeOutputDeltas(
      this.cost.gradient(this.outputLayer.activations, target),
    );
    for (let i = this.layers.length - 2; i >= 1; i--) {
      this.layers[i].computeHiddenDeltas();
    }
    for (const layer of this.computedLayers) layer.applyGradient(learningRate);
  }

  evaluate(sample: Sample): number {
    return this.cost.evaluate(this.forward(sample.inputs), sample.target);
  }

  averageCost(samples: readonly Sample[]): number {
    let total = 0;
    for (const sample of samples) total += this.evaluate(sample);
    return total / samples.length;
  }

  toString(): string {
    const shape = this.layers.map(layer => layer.size).join(" -> ");
    const count = (n: number) => n.toLocaleString("en-US");
    return `${shape}   (${count(this.parameterCount)} parameters:` +
      ` ${count(this.weightCount)} weights, ${count(this.biasCount)} biases)`;
  }
}

// --- Dataset, ProgressBar and Trainer, unchanged from step 05 ---

class Dataset {
  constructor(
    readonly name: string,
    readonly samples: readonly Sample[],
  ) {}

  get size(): number {
    return this.samples.length;
  }

  get inputSize(): number {
    return this.samples[0].inputs.length;
  }

  get outputSize(): number {
    return this.samples[0].target.length;
  }

  at(iteration: number): Sample {
    return this.samples[iteration % this.samples.length];
  }

  epochOf(iteration: number): number {
    return Math.floor(iteration / this.samples.length) + 1;
  }

  shapeWith(hiddenSizes: number[]): number[] {
    return [this.inputSize, ...hiddenSizes, this.outputSize];
  }
}

class ProgressBar {
  private readonly history: number[] = [];

  constructor(
    private readonly width = 40,
    private readonly window = 100,
  ) {}

  record(score: number): void {
    this.history.push(score);
    if (this.history.length > this.window) this.history.shift();
  }

  get average(): number {
    if (this.history.length === 0) return 0;
    return this.history.reduce((a, b) => a + b, 0) / this.history.length;
  }

  render(suffix = ""): void {
    const ratio = this.average;
    const filled = Math.round(ratio * this.width);
    const bar = "█".repeat(filled) + "░".repeat(this.width - filled);
    const line = ` ${(ratio * 100).toFixed(1).padStart(5)}% ${bar}  ${suffix} `;
    const red = Math.round(200 - 150 * ratio);
    const green = Math.round(130 * ratio);

    readline.cursorTo(process.stdout, 0);
    readline.clearLine(process.stdout, 0);
    process.stdout.write(`\x1b[48;2;${red};${green};0m${line}\x1b[0m`);
  }

  finish(): void {
    process.stdout.write("\n");
  }
}

interface ProgressMetric {
  score(output: number[], sample: Sample): number;
  caption(output: number[], sample: Sample): string;
}

class OutputCloseness implements ProgressMetric {
  score(output: number[], sample: Sample): number {
    let total = 0;
    for (let i = 0; i < sample.target.length; i++) {
      total += 1 - Math.abs(sample.target[i] - output[i]);
    }
    return total / sample.target.length;
  }

  caption(output: number[], sample: Sample): string {
    const want = sample.target.map(v => v.toFixed(0)).join(" ");
    const got = output.map(v => v.toFixed(3)).join(" ");
    return `want [${want}]  got [${got}]`;
  }
}

/**
 * NEW: the metric that actually makes sense for ten classes.
 *
 * OutputCloseness would score a network that answers "0.0 everywhere"
 * at 90%, because nine of the ten targets really are 0. Useless. What we
 * care about here is whether the loudest output neuron is the right one -
 * so we score a flat 1 or 0 and let the sliding window average it into
 * "percent correct over the last hundred digits".
 */
class ClassificationAccuracy implements ProgressMetric {
  score(output: number[], sample: Sample): number {
    return argmax(output) === argmax(sample.target) ? 1 : 0;
  }

  caption(output: number[], sample: Sample): string {
    const got = argmax(output);
    return `want ${argmax(sample.target)}  got ${got} (${(output[got] * 100).toFixed(0)}%)`;
  }
}

interface TrainingOptions {
  readonly learningRate: number;
  readonly iterations: number;
  readonly refreshEvery?: number;
  readonly metric?: ProgressMetric;
}

/** A prime, so the bar never locks onto the same handful of samples. */
const DEFAULT_REFRESH = 97;

class Trainer {
  private readonly bar = new ProgressBar();

  constructor(
    private readonly network: Network,
    private readonly options: TrainingOptions,
  ) {}

  train(dataset: Dataset): void {
    const {
      learningRate,
      iterations,
      refreshEvery = DEFAULT_REFRESH,
      metric = new OutputCloseness(),
    } = this.options;

    for (let i = 0; i < iterations; i++) {
      const sample = dataset.at(i);
      const output = this.network.forward(sample.inputs);
      this.bar.record(metric.score(output, sample));
      this.network.backpropagate(sample.target, learningRate);

      if (i % refreshEvery === 0) {
        this.bar.render(`epoch ${dataset.epochOf(i)}  ${metric.caption(output, sample)}`);
      }
    }

    this.bar.render("done");
    this.bar.finish();
  }
}

// --- New: one labelled image ---

const SHADES = " .:-=+*#%@";

/**
 * A 28x28 grayscale image with a known label. The one-hot target is
 * built once in the constructor: the training loop reads it thousands
 * of times, so it has no business being recomputed.
 */
class Digit implements Sample {
  static readonly SIZE = 28;

  readonly target: number[];

  constructor(
    readonly label: number,
    readonly inputs: number[],
  ) {
    // "This is a 3" becomes [0,0,0,1,0,0,0,0,0,0] - one neuron per digit.
    this.target = Array.from({ length: 10 }, (_, i) => (i === label ? 1 : 0));
  }

  /** Draw the image in the terminal, one character per pixel. */
  toAscii(indent = "  "): string {
    const rows: string[] = [];
    for (let row = 0; row < Digit.SIZE; row++) {
      let line = indent;
      for (let column = 0; column < Digit.SIZE; column++) {
        const value = this.inputs[row * Digit.SIZE + column];
        line += SHADES[Math.min(Math.floor(value * SHADES.length), SHADES.length - 1)];
      }
      rows.push(line);
    }
    return rows.join("\n");
  }
}

// --- New: reading the raw MNIST files ---

/**
 * MNIST ships in the IDX format: a short big-endian header, then the raw
 * bytes back to back. No compression, no delimiters. We only decode the
 * images we actually keep - decoding all 60000 would cost far more time
 * and memory than the training itself.
 */
class MnistLoader {
  constructor(private readonly directory: string) {}

  get isAvailable(): boolean {
    return fs.existsSync(path.join(this.directory, "train-images-idx3-ubyte"));
  }

  /** Load `count` digits, with the ten classes evenly represented. */
  load(kind: "train" | "test", count: number): Digit[] {
    const prefix = kind === "train" ? "train" : "t10k";
    const labels = this.readLabels(`${prefix}-labels-idx1-ubyte`);
    const images = fs.readFileSync(path.join(this.directory, `${prefix}-images-idx3-ubyte`));

    return this.pickBalanced(labels, count).map(
      index => new Digit(labels[index], this.decodeImage(images, index)),
    );
  }

  private readLabels(file: string): number[] {
    const buffer = fs.readFileSync(path.join(this.directory, file));
    const count = buffer.readUInt32BE(4);
    const labels: number[] = [];
    for (let i = 0; i < count; i++) labels.push(buffer[8 + i]);
    return labels;
  }

  private decodeImage(buffer: Buffer, index: number): number[] {
    const rows = buffer.readUInt32BE(8);
    const columns = buffer.readUInt32BE(12);
    const pixels = rows * columns;
    const start = 16 + index * pixels;

    const image: number[] = new Array(pixels);
    // A byte is 0-255; neurons want 0-1.
    for (let i = 0; i < pixels; i++) image[i] = buffer[start + i] / 255;
    return image;
  }

  /**
   * Take the same number of each digit, then shuffle.
   * Two separate reasons, both of which bite at small sample counts:
   * a network fed 30 threes and 12 fives learns that threes are simply
   * more likely than fives, and a network fed a hundred fours in a row
   * forgets everything else while it does so.
   */
  private pickBalanced(labels: readonly number[], count: number): number[] {
    const perClass = Math.ceil(count / 10);
    const buckets: number[][] = Array.from({ length: 10 }, () => []);

    for (let i = 0; i < labels.length; i++) {
      const bucket = buckets[labels[i]];
      if (bucket.length < perClass) bucket.push(i);
    }

    const chosen = buckets.flat();
    for (let i = chosen.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [chosen[i], chosen[j]] = [chosen[j], chosen[i]];
    }
    return chosen.slice(0, count);
  }
}

// --- New: turning 10 activations into an answer ---

interface Prediction {
  readonly digit: number;
  readonly confidence: number;
}

class Classifier {
  constructor(private readonly network: Network) {}

  predict(digit: Digit): Prediction {
    const output = this.network.forward(digit.inputs);
    const best = argmax(output);
    return { digit: best, confidence: output[best] };
  }

  evaluate(digits: readonly Digit[]): Evaluation {
    const evaluation = new Evaluation();
    for (const digit of digits) {
      evaluation.record(digit.label, this.predict(digit).digit);
    }
    return evaluation;
  }
}

/** The scoreboard. Overall accuracy hides which digits the network confuses. */
class Evaluation {
  private readonly correct = new Array(10).fill(0);
  private readonly total = new Array(10).fill(0);

  record(actual: number, predicted: number): void {
    this.total[actual]++;
    if (actual === predicted) this.correct[actual]++;
  }

  get sampleCount(): number {
    return this.total.reduce((a, b) => a + b, 0);
  }

  get correctCount(): number {
    return this.correct.reduce((a, b) => a + b, 0);
  }

  get accuracy(): number {
    return this.sampleCount === 0 ? 0 : this.correctCount / this.sampleCount;
  }

  toString(): string {
    const header =
      `  Overall: ${this.correctCount}/${this.sampleCount}` +
      `  (${(this.accuracy * 100).toFixed(1)}%)\n`;

    const rows = this.total.map((total, digit) => {
      const ratio = total === 0 ? 0 : this.correct[digit] / total;
      const bar = "█".repeat(Math.round(ratio * 20)).padEnd(20, "░");
      const percent = total === 0 ? "  -" : `${(ratio * 100).toFixed(0).padStart(3)}%`;
      return `    ${digit}  ${percent}  ${bar}  (${this.correct[digit]}/${total})`;
    });

    return header + "\n" + rows.join("\n");
  }
}

// --- Go ---

const MNIST_DIR = path.join(__dirname, "..", "data", "mnist");
const loader = new MnistLoader(MNIST_DIR);

if (!loader.isAvailable) {
  console.error("MNIST data not found. Run:  npm run download-mnist");
  process.exit(1);
}

const trainCount = parseInt(process.argv[2] || "1000", 10);
const iterations = parseInt(process.argv[3] || "50000", 10);
const learningRate = parseFloat(process.argv[4] || "0.5");

const training = new Dataset("MNIST", loader.load("train", trainCount));
const testing = loader.load("test", 500);

const network = new Network(training.shapeWith([128]));
const classifier = new Classifier(network);

console.log("=== MNIST handwritten digits ===\n");
console.log(`  Network      ${network}`);
console.log(`  Training     ${training.size} digits, ${iterations} iterations, lr ${learningRate}`);
console.log(`  Testing      ${testing.length} digits the network will never train on\n`);

console.log(`Before training: ${(classifier.evaluate(testing).accuracy * 100).toFixed(1)}% (random guessing is 10%)\n`);

new Trainer(network, {
  learningRate,
  iterations,
  metric: new ClassificationAccuracy(),
}).train(training);

console.log("\n=== Test set: digits it has never seen ===\n");
console.log(classifier.evaluate(testing).toString());

// --- Look at a few, including the mistakes ---

console.log("\n=== A few predictions ===\n");

const hits = testing.filter(d => classifier.predict(d).digit === d.label);
const misses = testing.filter(d => classifier.predict(d).digit !== d.label);
const shown = [...hits.slice(0, 2), ...misses.slice(0, 2)];

for (const digit of shown) {
  const prediction = classifier.predict(digit);
  const verdict = prediction.digit === digit.label ? "ok" : "MISS";
  console.log(
    `  actual ${digit.label}   predicted ${prediction.digit}` +
    `   confidence ${(prediction.confidence * 100).toFixed(0)}%   ${verdict}`,
  );
  console.log(digit.toAscii());
  console.log("");
}

console.log("--- Key insight ---");
console.log("Not one line of the network changed between XOR and this.");
console.log("Same Neuron, same Layer, same backpropagate(). The only");
console.log("difference is 784 inputs instead of 2, and enough hidden");
console.log("neurons to hold the features it needs to invent.");
if (misses.length > 0) {
  console.log("");
  console.log("Look at the misses above and ask whether you would have been sure.");
  console.log("The network's mistakes cluster on genuinely ambiguous handwriting,");
  console.log("which is a different thing from being confused at random.");
}

console.log("\n--- Try it ---");
console.log("npx tsx steps/06-digits.ts 500 20000       fast, roughly 80%");
console.log("npx tsx steps/06-digits.ts 2000 100000     slower, roughly 90%");
console.log("Then change [128] to [32] in shapeWith - how far can you shrink it");
console.log("before the accuracy falls apart?");

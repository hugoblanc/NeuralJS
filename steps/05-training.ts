// Step 05 - The Training Loop
// ===========================
// The network can already learn - step 04 proved it. What is missing is
// everything around the learning: which samples, for how long, at what
// speed, and how do we watch it happen.
//
// That is deliberately NOT the network's job. A Network knows how to go
// forward and backward; it should not know about epochs, progress bars
// or command line arguments. So we give that work to its own objects:
//
//   Dataset         - the samples, and how to cycle through them
//   ProgressBar     - how training looks on screen
//   ProgressMetric  - what "doing well" means for this problem
//   Trainer         - the loop: forward, measure, backpropagate, repeat
//
// This is the boring kind of OOP, and it is the kind that pays off:
// step 06 swaps the Dataset for handwritten digits and plugs in a
// different ProgressMetric, and the Network class does not change by
// a single character.
//
// NEW IN THIS STEP
//   class Dataset, class ProgressBar, class Trainer
//   interface ProgressMetric, class OutputCloseness
//
// 3Blue1Brown reference: Chapter 2, 6:00-15:00
// https://www.youtube.com/watch?v=IHZwWFHWa-w

import * as readline from "readline";

function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

function sigmoidDerivative(x: number): number {
  const s = sigmoid(x);
  return s * (1 - s);
}

// --- Everything from steps 01-04 ---

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

// --- New: the data ---

/**
 * A named collection of samples. It also knows the shape of the network
 * that can consume it, which spares us from hardcoding sizes everywhere.
 */
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

  /** Cycle forever through the samples. Iteration 5 of 4 samples is sample 1. */
  at(iteration: number): Sample {
    return this.samples[iteration % this.samples.length];
  }

  /** One pass over every sample is an epoch. */
  epochOf(iteration: number): number {
    return Math.floor(iteration / this.samples.length) + 1;
  }

  /** The input and output sizes are fixed by the data - only the middle is ours. */
  shapeWith(hiddenSizes: number[]): number[] {
    return [this.inputSize, ...hiddenSizes, this.outputSize];
  }
}

// --- New: the display ---

/**
 * A live progress bar. It smooths the metric over a sliding window,
 * because the score on a single sample tells you almost nothing.
 */
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

    // Red when bad, green when good.
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

// --- New: the loop ---

/**
 * What the progress bar measures. This is NOT the cost function: the cost
 * exists to produce gradients, this exists to be read by a human. Keeping
 * them apart means step 06 can measure "did it name the right digit"
 * without touching a line of the maths.
 */
interface ProgressMetric {
  /** 1 is perfect, 0 is as wrong as possible. */
  score(output: number[], sample: Sample): number;
  /** The text shown next to the bar. */
  caption(output: number[], sample: Sample): string;
}

/** How close are the raw output numbers to the targets? */
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

interface TrainingOptions {
  readonly learningRate: number;
  readonly iterations: number;
  /** Redrawing the bar costs more than training does. Only do it now and then. */
  readonly refreshEvery?: number;
  readonly metric?: ProgressMetric;
}

/**
 * The default refresh stride. It is a prime on purpose: refreshing every
 * 100 iterations on a 4-sample dataset would only ever show you sample 0,
 * and you would spend a while wondering why the network never sees a 1.
 */
const DEFAULT_REFRESH = 97;

/**
 * Drives a network through a dataset. This is stochastic gradient descent
 * with a batch size of one: every single sample updates the weights.
 */
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

      // 1. Forward - what does the network currently think?
      const output = this.network.forward(sample.inputs);

      // 2. Measure, for us. The network itself never reads this number.
      this.bar.record(metric.score(output, sample));

      // 3. Backward - nudge every weight and bias downhill.
      this.network.backpropagate(sample.target, learningRate);

      if (i % refreshEvery === 0) {
        this.bar.render(`epoch ${dataset.epochOf(i)}  ${metric.caption(output, sample)}`);
      }
    }

    this.bar.render("done");
    this.bar.finish();
  }
}

// --- The problems we can throw at it ---

/** Add two 2-bit numbers. The answer needs 3 bits, so it must learn a carry. */
function additionSamples(): Sample[] {
  const samples: Sample[] = [];
  for (let a = 0; a < 4; a++) {
    for (let b = 0; b < 4; b++) {
      const sum = a + b;
      samples.push({
        inputs: [(a >> 1) & 1, a & 1, (b >> 1) & 1, b & 1],
        target: [(sum >> 2) & 1, (sum >> 1) & 1, sum & 1],
      });
    }
  }
  return samples;
}

const DATASETS: Record<string, Dataset> = {
  xor: new Dataset("XOR gate", [
    { inputs: [0, 0], target: [0] },
    { inputs: [0, 1], target: [1] },
    { inputs: [1, 0], target: [1] },
    { inputs: [1, 1], target: [0] },
  ]),

  // Output 1 when an odd number of bits are set - XOR, generalised to 4 bits.
  parity: new Dataset("Parity of 4 bits", [
    { inputs: [0, 0, 0, 0], target: [0] }, { inputs: [1, 0, 0, 0], target: [1] },
    { inputs: [0, 1, 0, 0], target: [1] }, { inputs: [0, 0, 1, 0], target: [1] },
    { inputs: [0, 0, 0, 1], target: [1] }, { inputs: [1, 1, 0, 0], target: [0] },
    { inputs: [1, 0, 1, 0], target: [0] }, { inputs: [1, 0, 0, 1], target: [0] },
    { inputs: [0, 1, 1, 0], target: [0] }, { inputs: [0, 1, 0, 1], target: [0] },
    { inputs: [0, 0, 1, 1], target: [0] }, { inputs: [1, 1, 1, 0], target: [1] },
    { inputs: [1, 1, 0, 1], target: [1] }, { inputs: [1, 0, 1, 1], target: [1] },
    { inputs: [0, 1, 1, 1], target: [1] }, { inputs: [1, 1, 1, 1], target: [0] },
  ]),

  add: new Dataset("2-bit addition", additionSamples()),
};

// --- Go ---

const datasetName = process.argv[2] || "xor";
const learningRate = parseFloat(process.argv[3] || "0.5");
const iterations = parseInt(process.argv[4] || "30000", 10);

const dataset = DATASETS[datasetName];
if (!dataset) {
  console.error(`Unknown dataset "${datasetName}". Try: ${Object.keys(DATASETS).join(", ")}`);
  process.exit(1);
}

const network = new Network(dataset.shapeWith([8]));

console.log(`=== ${dataset.name} ===`);
console.log(`Network ${network}  |  lr ${learningRate}  |  ${iterations} iterations  |  ${dataset.size} samples\n`);
console.log(`Cost before training: ${network.averageCost(dataset.samples).toFixed(4)}\n`);

new Trainer(network, { learningRate, iterations }).train(dataset);

console.log(`\nCost after training:  ${network.averageCost(dataset.samples).toFixed(6)}\n`);
console.log("=== Final answers ===\n");

let correct = 0;
for (const sample of dataset.samples) {
  const output = network.forward(sample.inputs);
  const rounded = output.map(v => (v > 0.5 ? 1 : 0));
  const ok = rounded.every((v, i) => v === sample.target[i]);
  if (ok) correct++;
  console.log(
    `  [${sample.inputs.join(" ")}]  ->  [${output.map(v => v.toFixed(3)).join(" ")}]` +
    `   want [${sample.target.join(" ")}]  ${ok ? "ok" : "MISS"}`,
  );
}
console.log(`\n  ${correct}/${dataset.size} correct\n`);

console.log("--- Key insight ---");
console.log("The learning rate is the size of each downhill step, and there is");
console.log("no universally correct value. Run the sweep below before assuming");
console.log("you know what 'too big' means: on XOR, a rate of 8 beats the");
console.log("default, and things only fall apart somewhere past 20. A rate that");
console.log("is reckless on one problem is timid on another - which is why");
console.log("finding it is half the job.");

console.log("\n--- Try it ---");
console.log("npx tsx steps/05-training.ts xor 0.5         default, converges cleanly");
console.log("npx tsx steps/05-training.ts xor 0.005       too slow: 30000 updates is not enough");
console.log("npx tsx steps/05-training.ts xor 8.0         surprisingly good - try to predict this first");
console.log("npx tsx steps/05-training.ts xor 30          broken: it lands at cost 0.5 and stays");
console.log("npx tsx steps/05-training.ts parity 0.5      harder: 4-bit parity");
console.log("npx tsx steps/05-training.ts add 0.5 60000   hardest: it must invent a carry");
console.log("");
console.log("Results move with the random initialisation, so run each one twice.");

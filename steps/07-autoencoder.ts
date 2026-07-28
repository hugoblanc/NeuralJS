// Step 07 - The Autoencoder
// =========================
// One last idea, and the strangest one: train a network to output
// exactly what it was given. Copy the input. That sounds useless -
// until you force the data through a narrow layer on the way.
//
//   8 inputs -> 5 neurons -> 3 neurons -> 5 neurons -> 8 outputs
//                            ^^^^^^^^^
//                            everything must fit through here
//
// There are 8 possible inputs and only 3 neurons in the middle. To
// rebuild the input on the far side, the network has to compress "which
// of the 8" into 3 numbers, with nobody telling it how.
//
// The obvious guess is that it invents binary: 2^3 = 8, exactly enough.
// Run it. The answer is no, and why it is no turns out to be the whole
// point of the step.
//
// This is also where inheritance finally earns its place. An
// autoencoder IS a network - same forward, same backpropagate, nothing
// overridden. What it adds is a vocabulary: encode(), decode(), and the
// notion of a bottleneck. So it extends Network rather than reimplementing
// or wrapping it.
//
// NEW IN THIS STEP
//   class Autoencoder extends Network  -  encode(), decode(), bottleneckLayer
//   class ExactReconstruction implements ProgressMetric

import * as readline from "readline";

function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

function sigmoidDerivative(x: number): number {
  const s = sigmoid(x);
  return s * (1 - s);
}

// --- Everything from steps 01-05, unchanged ---

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
 * NEW: all or nothing. Round every output to 0 or 1 and ask whether the
 * pattern came back exactly. Half a reconstruction is not a reconstruction.
 */
class ExactReconstruction implements ProgressMetric {
  score(output: number[], sample: Sample): number {
    return output.every((value, i) => Math.round(value) === sample.target[i]) ? 1 : 0;
  }

  caption(output: number[], sample: Sample): string {
    const want = sample.target.join("");
    const got = output.map(v => Math.round(v)).join("");
    return `want ${want}  got ${got}`;
  }
}

interface TrainingOptions {
  readonly learningRate: number;
  readonly iterations: number;
  readonly refreshEvery?: number;
  readonly metric?: ProgressMetric;
}

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

// --- New: a network that must rebuild its own input ---

/**
 * An autoencoder is a network shaped like an hourglass, trained to
 * reproduce its input. Nothing about the maths changes - only the shape
 * of the network and the choice of target. So we inherit all of it and
 * add three things: where the bottleneck is, and how to use each half
 * of the network on its own.
 */
class Autoencoder extends Network {
  constructor(
    readonly patternSize: number,
    hiddenSize: number,
    readonly codeSize: number,
  ) {
    super([patternSize, hiddenSize, codeSize, hiddenSize, patternSize]);
  }

  private get bottleneckIndex(): number {
    return Math.floor(this.layers.length / 2);
  }

  get bottleneckLayer(): Layer {
    return this.layers[this.bottleneckIndex];
  }

  /** First half: squeeze a pattern down to `codeSize` numbers. */
  encode(inputs: number[]): number[] {
    this.forward(inputs);
    return this.bottleneckLayer.activations;
  }

  /**
   * Second half: hand the bottleneck a code and let the rest of the
   * network run. Note that we can feed it a code the encoder has never
   * produced - which is how we find out whether it learned a real code
   * or just memorised eight paths.
   */
  decode(code: number[]): number[] {
    this.bottleneckLayer.receive(code);
    for (const layer of this.layers.slice(this.bottleneckIndex + 1)) {
      layer.activate();
    }
    return this.outputLayer.activations;
  }

  /**
   * The 2^codeSize corners of the code cube: the only codes the bottleneck
   * could produce if it really were binary. It is not, so these are just
   * 2^codeSize convenient points out of infinitely many.
   */
  get cornerCodes(): number[][] {
    const total = 2 ** this.codeSize;
    return Array.from({ length: total }, (_, n) =>
      Array.from({ length: this.codeSize }, (_, bit) => (n >> (this.codeSize - 1 - bit)) & 1),
    );
  }
}

/** The whole dataset: eight patterns, each its own target. */
class Reconstruction implements Sample {
  constructor(readonly inputs: number[]) {}

  /** The target IS the input. That single line is what makes it an autoencoder. */
  get target(): number[] {
    return this.inputs;
  }
}

// --- Go ---

const PATTERN_SIZE = 8;
const CODE_SIZE = 3;

const learningRate = parseFloat(process.argv[2] || "1.0");
const iterations = parseInt(process.argv[3] || "80000", 10);

const patterns = Array.from({ length: PATTERN_SIZE }, (_, i) =>
  new Reconstruction(Array.from({ length: PATTERN_SIZE }, (_, j) => (i === j ? 1 : 0))),
);
const dataset = new Dataset("one-hot patterns", patterns);

const autoencoder = new Autoencoder(PATTERN_SIZE, 5, CODE_SIZE);

console.log("=== Autoencoder ===\n");
console.log(`  Network      ${autoencoder}`);
console.log(`  Bottleneck   ${autoencoder.codeSize} neurons for ${dataset.size} patterns`);
console.log(`  Training     ${iterations} iterations, lr ${learningRate}\n`);

new Trainer(autoencoder, {
  learningRate,
  iterations,
  metric: new ExactReconstruction(),
}).train(dataset);

console.log(`\nFinal cost: ${autoencoder.averageCost(patterns).toFixed(6)}`);

// --- What code did it invent? ---

const codes = patterns.map(pattern => autoencoder.encode(pattern.inputs));
const rounded = codes.map(code => code.map(v => Math.round(v)).join(""));

console.log("\n=== The encoder: pattern -> code ===\n");

let recovered = 0;

patterns.forEach((pattern, i) => {
  const raw = codes[i].map(v => v.toFixed(2)).join(", ");
  autoencoder.forward(pattern.inputs);
  const rebuilt = autoencoder.outputLayer.activations.map(v => Math.round(v)).join("");
  const ok = rebuilt === pattern.target.join("");
  if (ok) recovered++;
  const collides = rounded.filter(code => code === rounded[i]).length > 1;

  console.log(
    `  pattern ${i}  ${pattern.inputs.join("")}` +
    `  ->  [${raw}]  ~  ${rounded[i]}` +
    `  ->  ${rebuilt}  ${ok ? "ok" : "MISS"}` +
    (collides ? "   <- shares its rounded code" : ""),
  );
});

// How close do the two nearest codes sit? If two patterns were mapped to
// essentially the same point, the decoder has no way to tell them apart.
let closest = Infinity;
let closestPair = [0, 0];
for (let i = 0; i < codes.length; i++) {
  for (let j = i + 1; j < codes.length; j++) {
    const distance = Math.hypot(...codes[i].map((v, k) => v - codes[j][k]));
    if (distance < closest) {
      closest = distance;
      closestPair = [i, j];
    }
  }
}

console.log(`\n  ${recovered} of ${patterns.length} patterns came back intact.`);
if (recovered < patterns.length) {
  console.log(`  ${patterns.length - recovered} did not, so this run did not fully solve the problem.`);
  console.log(
    closest < 0.1
      ? `  Patterns ${closestPair[0]} and ${closestPair[1]} were mapped to the same point, and the decoder` +
        "\n  cannot separate what the encoder merged."
      : "  The codes are all distinct, so this is under-training rather than a\n  collapse: the decoder has not finished learning to read them.",
  );
  console.log("  Either way, run it again or train longer. The rest still holds.");
}

// The decisive one, and it appears on its own when you shrink CODE_SIZE:
// a binary code of n bits has 2^n slots, so recovering more patterns than
// that settles the question without any interpretation.
if (recovered > 2 ** CODE_SIZE) {
  console.log(`\n  Note that ${recovered} patterns came through a bottleneck with only`);
  console.log(`  ${2 ** CODE_SIZE} binary codes available. Whatever it is storing in there, it is`);
  console.log(`  not bits: no ${CODE_SIZE}-bit code could have carried them.`);
}

// --- Then what did it invent? ---

// A neuron parked in the middle of its range is carrying a value, not a bit.
// This is the sound evidence: the count of distinct rounded codes cannot
// exceed 2^CODE_SIZE, so a low number there would prove nothing on its own.
const activations = codes.flat();
const undecided = activations.filter(v => v > 0.15 && v < 0.85).length;
const distinct = new Set(rounded).size;

console.log(`\n  ${undecided} of the ${activations.length} bottleneck activations sit nowhere near 0 or 1.`);

// Rounding can never yield more than 2^CODE_SIZE distinct codes, so only say
// something when it loses patterns the corners could actually have held.
if (distinct < Math.min(patterns.length, 2 ** CODE_SIZE)) {
  console.log(`  And rounding to bits collapses the ${patterns.length} codes down to ${distinct}, fewer even`);
  console.log(`  than the ${2 ** CODE_SIZE} corners available, so the bits alone could not tell them apart.`);
}
console.log(`  Whatever the network built in there, it is not a ${CODE_SIZE}-bit word.`);

console.log("\n=== Why not binary? ===\n");
console.log("  Each bottleneck neuron outputs a real number in [0, 1], so a code is a");
console.log(`  point in ${CODE_SIZE}-dimensional space, not a ${CODE_SIZE}-bit word. The all-0-or-1`);
console.log(`  corners are just ${2 ** CODE_SIZE} of the infinitely many points available, so nothing`);
console.log(`  pushes the network towards them. It only needs its ${PATTERN_SIZE} points far`);
console.log("  enough apart for the decoder to tell them apart - and 'far enough'");
console.log("  is a distance, not a bit.\n");

console.log(`  Closest two codes: patterns ${closestPair[0]} and ${closestPair[1]}, ${closest.toFixed(3)} apart,`);
console.log(`  in a space whose longest diagonal is ${Math.sqrt(CODE_SIZE).toFixed(3)}.`);
console.log(
  closest < 0.1
    ? "  That is a collapse: those two are effectively the same point."
    : "  Plenty of room - the decoder never has to guess.",
);

// --- The decoder, fed codes it has never seen ---

const produced = new Set(rounded);
const corners = autoencoder.cornerCodes;
const unvisited = corners.filter(code => !produced.has(code.join(""))).length;

console.log("\n=== The decoder: corner codes -> pattern ===\n");
console.log(`  Feeding the bottleneck all ${corners.length} pure binary codes directly,`);
console.log(`  bypassing the encoder. ${unvisited} of them it never produces itself.\n`);

for (const code of corners) {
  const output = autoencoder.decode(code);
  const rebuilt: number[] = output.map(v => (v > 0.5 ? 1 : 0));
  const lit = rebuilt.reduce((a, b) => a + b, 0);
  const verdict = lit === 1 ? `pattern ${rebuilt.indexOf(1)}` : lit === 0 ? "nothing" : "several at once";
  const note = produced.has(code.join("")) ? "" : "   <- encoder never lands here";

  console.log(`  code ${code.join("")}  ->  ${rebuilt.join("")}  = ${verdict}${note}`);
}

if (unvisited > 0) {
  console.log("\n  The decoder has an opinion even about corners the encoder never visits.");
} else {
  console.log("\n  The decoder answers for every corner, and it would answer just as");
  console.log("  readily for any point in between.");
}
console.log("  It is a continuous function over the whole cube, not a lookup table");
console.log(`  with ${corners.length} entries - which is why the space between codes means something.`);

console.log("\n--- Key insight ---");
if (recovered === patterns.length) {
  console.log(`The compression is real: ${patterns.length} patterns survive a ${CODE_SIZE}-neuron waist.`);
} else {
  console.log(`This run only got ${recovered} of ${patterns.length} patterns through the ${CODE_SIZE}-neuron waist,`);
  console.log("but the shape of what it learned is the same either way.");
}
console.log("The code the network chose is continuous, not symbolic.");
console.log("That is the whole idea behind embeddings: a learned representation");
console.log("is a position in a space, not an entry in a dictionary. Distance");
console.log("in that space is what carries meaning.");
console.log("");
console.log("The hidden layers of step 06 were doing the same thing quietly:");
console.log("inventing whatever intermediate features the problem demanded,");
console.log("with no instruction about what those features should be.");

console.log("\n--- Try it ---");
console.log("Set CODE_SIZE = 2 and run it a few times. If the bottleneck really");
console.log("were binary, 2 neurons would give 4 codes for 8 patterns and it");
console.log("could never recover more than 4. It gets all 8 more often than not -");
console.log("roughly 6 runs in 10, so you will see both outcomes. Even the runs");
console.log("that fall short beat the 4 that a 2-bit code would cap you at.");
console.log("");
console.log("Then set CODE_SIZE = 1: 8 patterns strung out along a single line.");
console.log("This one does break, every time - around 1 to 4 patterns survive.");
console.log("Not because it ran out of codes, but because it ran out of room to");
console.log("keep them apart.");

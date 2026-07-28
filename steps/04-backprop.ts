// Step 04 - Backpropagation
// =========================
// We can measure the cost. Now we reduce it.
//
// The idea: work out how much each weight contributed to the error,
// then nudge it the other way. The error has to travel from the output
// back to the input - which is exactly why we need one new thing in the
// object graph.
//
// Until now a Synapse only knew where it read FROM. The forward pass
// never needs more than that. But blame flows the other way: a hidden
// neuron can only know how wrong it is by asking the neurons it FEEDS.
// So a synapse now knows both ends, and a neuron keeps both lists:
//
//   neuron.inputs   ->  used by forward()          (who do I read from?)
//   neuron.outputs  ->  used by backpropagate()    (who do I feed?)
//
// Every neuron then stores one new number, its delta:
//
//   delta = "if my raw sum went up by 1, the cost would go up by this much"
//
// Once a neuron knows its delta, it has three levers, all local:
//   1. nudge its bias against its delta
//   2. nudge its incoming weights against its delta, in proportion to
//      what each source actually sent
//   3. pass the blame back to those sources
//
// "Against its delta" and not "down": a neuron whose delta is negative is
// one the cost wants LOUDER, so its bias and weights go up.
//
// No neuron ever sees the whole network. That is the magic.
//
// NEW IN THIS STEP
//   Synapse gains `target`; Neuron gains `outputs` and `delta`
//   Neuron   -  computeOutputDelta(), computeHiddenDelta(), applyGradient()
//   Layer    -  the same three, applied to all its neurons
//   Network  -  backpropagate()
//
// 3Blue1Brown reference: Chapter 3
// https://www.youtube.com/watch?v=Ilg3gGewQ5U

function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

/** How much does the activation move when the raw sum moves? */
function sigmoidDerivative(x: number): number {
  const s = sigmoid(x);
  return s * (1 - s);
}

// --- Synapse: now knows both ends ---

class Synapse {
  weight: number;

  constructor(
    readonly source: Neuron,
    readonly target: Neuron,
    weight: number = Math.random() * 2 - 1,
  ) {
    this.weight = weight;
  }

  /** Forward: what this connection sends. */
  get contribution(): number {
    return this.source.activation * this.weight;
  }

  /** Backward: how much of the target's blame belongs to the source. */
  get blame(): number {
    return this.weight * this.target.delta;
  }
}

// --- Neuron: now has a backward half ---

class Neuron {
  bias = 1;
  sum = 0;
  activation = 0;

  /** dCost/dSum for this neuron. Filled in during backpropagation. */
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

  /**
   * Output neuron: the cost function tells us directly how wrong we are.
   * Chain rule - dCost/dSum = dCost/dActivation * dActivation/dSum.
   */
  computeOutputDelta(costGradient: number): void {
    this.delta = costGradient * sigmoidDerivative(this.sum);
  }

  /**
   * Hidden neuron: nobody tells us anything. We collect the blame from
   * the neurons we feed, weighted by how strongly we feed them.
   * Same chain rule, one link further back.
   */
  computeHiddenDelta(): void {
    let error = 0;
    for (const synapse of this.outputs) error += synapse.blame;
    this.delta = error * sigmoidDerivative(this.sum);
  }

  /** Step downhill: adjust my bias and my incoming weights. */
  applyGradient(learningRate: number): void {
    this.bias -= this.delta * learningRate;
    for (const synapse of this.inputs) {
      // A weight is blamed in proportion to what its source actually sent:
      // a source that was silent (activation 0) cannot be at fault.
      synapse.weight -= this.delta * synapse.source.activation * learningRate;
    }
  }
}

// --- Layer: same three moves, for all its neurons ---

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

// --- Cost function, unchanged from step 03 ---

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

// --- Network: forward, and now backward ---

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

  /**
   * Backpropagation, in the same shape as forward() - just reversed.
   * Call it right after forward(): it reads the activations and sums
   * that the forward pass left behind on every neuron.
   */
  backpropagate(target: number[], learningRate: number): void {
    // 1. The output layer is the only one the cost function can talk to.
    this.outputLayer.computeOutputDeltas(
      this.cost.gradient(this.outputLayer.activations, target),
    );

    // 2. Blame travels right to left. Order matters: a layer can only
    //    compute its deltas once the layer it feeds already has its own.
    for (let i = this.layers.length - 2; i >= 1; i--) {
      this.layers[i].computeHiddenDeltas();
    }

    // 3. Now every neuron knows its delta, so every neuron can fix itself.
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

// --- Watch it learn XOR ---

const xor: Sample[] = [
  { inputs: [0, 0], target: [0] },
  { inputs: [0, 1], target: [1] },
  { inputs: [1, 0], target: [1] },
  { inputs: [1, 1], target: [0] },
];

const net = new Network([2, 8, 1]);
const learningRate = 0.5;
const iterations = 10000;

// --- First, one backward pass in slow motion ---

console.log("=== A single backward pass, before any training ===\n");
console.log("  Feeding [1, 0], which should output 1.");
console.log("  A learning rate of 0 means: compute the deltas, change nothing.\n");

net.forward([1, 0]);
net.backpropagate([1], 0);

const deltas = net.computedLayers.flatMap(layer => layer.neurons.map(n => Math.abs(n.delta)));
const largest = Math.max(...deltas);

net.computedLayers.forEach((layer, index) => {
  const kind = layer === net.outputLayer ? "output" : "hidden";
  console.log(`  Layer ${index + 1} (${kind}):`);
  for (const [i, neuron] of layer.neurons.entries()) {
    const bar = "█".repeat(Math.round((Math.abs(neuron.delta) / largest) * 30));
    console.log(
      `    neuron ${i}  activation ${neuron.activation.toFixed(3)}` +
      `  delta ${neuron.delta >= 0 ? " " : ""}${neuron.delta.toFixed(5)}  ${bar}`,
    );
  }
});

console.log("\n  The output neuron carries the biggest delta - it is the one the");
console.log("  cost function speaks to directly. What reaches the hidden layer is");
console.log("  a fraction of it, split according to the weights in between.");
console.log("  Deeper networks make that fraction smaller and smaller: that is");
console.log("  the vanishing gradient problem, already visible across one hidden layer.");

// --- Now let it learn ---

console.log("\n=== Learning XOR ===\n");
console.log(`  ${net}`);
console.log(`  learning rate ${learningRate}, ${iterations} updates`);
console.log(`  (one update per sample, so ${iterations / xor.length} passes over the 4 samples)\n`);
console.log("  update     cost   [0,0]   [0,1]   [1,0]   [1,1]");
console.log("  " + "-".repeat(47));

function reportRow(update: number): void {
  const cost = net.averageCost(xor);
  const outputs = xor.map(s => net.forward(s.inputs)[0].toFixed(3)).join("   ");
  console.log(`  ${update.toString().padStart(5)}  ${cost.toFixed(6)}   ${outputs}`);
}

for (let i = 0; i < iterations; i++) {
  if (i % 1000 === 0) reportRow(i);

  const sample = xor[i % xor.length];
  net.forward(sample.inputs);
  net.backpropagate(sample.target, learningRate);
}
reportRow(iterations);

console.log("\n=== Final answers ===\n");
for (const sample of xor) {
  const [output] = net.forward(sample.inputs);
  const correct = Math.abs(output - sample.target[0]) < 0.1;
  console.log(
    `  [${sample.inputs.join(", ")}]  ->  ${output.toFixed(4)}` +
    `   (want ${sample.target[0]})  ${correct ? "ok" : "MISS"}`,
  );
}

console.log("\n--- Key insight ---");
console.log("Backpropagation is the chain rule, applied one layer at a time.");
console.log("Each neuron only ever talks to its immediate neighbours:");
console.log("it reads blame from the neurons it feeds, and passes blame");
console.log("to the neurons it reads from. Nothing coordinates them.");
console.log("Stack that on a million neurons and you have deep learning.");

console.log("\n--- Try it ---");
console.log("Raise the learning rate to 5.0 - it converges better, not worse.");
console.log("Raise it to 50 and it dies: every neuron saturates on the first few");
console.log("huge steps, the sigmoid slope goes to zero, and the network answers");
console.log("0.000 to everything forever. Not too fast - permanently stuck.");
console.log("There is no single threshold where 'too big' begins.");
console.log("Shrink the hidden layer to 1 neuron: new Network([2, 1, 1]).");
console.log("It gets stuck around 1 answer out of 4, and no amount of training");
console.log("helps: with a single hidden neuron the whole network is a monotonic");
console.log("function of one weighted sum of the inputs, and no such function");
console.log("can be low at [0,0] and [1,1] while being high at [0,1] and [1,0].");
console.log("The width of a layer is not a detail - it bounds what is expressible.");

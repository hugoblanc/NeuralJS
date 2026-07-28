// Step 03 - The Cost Function
// ===========================
// How do you tell a network it is doing a bad job?
// You need one number that says "this is how wrong you are".
// That number is the cost.
//
// We make it an object rather than a loose function, because a cost
// function has two faces that must always agree:
//
//   evaluate()  - how wrong am I?          (a number, for us to read)
//   gradient()  - which way is MORE wrong? (a vector, for step 04)
//
// The gradient points uphill, towards higher cost. That is not a mistake:
// step 04 subtracts it, and the minus sign in that subtraction is the
// entire idea of gradient descent.
//
// Keeping them in one class means you can swap the whole thing later
// (cross-entropy instead of squared error) without touching the network.
//
// NEW IN THIS STEP
//   interface Sample, interface CostFunction
//   class MeanSquaredError
//   Network gains a cost function, evaluate() and averageCost()
//
// 3Blue1Brown reference: Chapter 2, 0:00-6:00
// https://www.youtube.com/watch?v=IHZwWFHWa-w

function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

// --- Same as step 02 ---

class Synapse {
  weight: number;

  constructor(
    readonly source: Neuron,
    weight: number = Math.random() * 2 - 1,
  ) {
    this.weight = weight;
  }

  get contribution(): number {
    return this.source.activation * this.weight;
  }
}

class Neuron {
  bias = 1;
  sum = 0;
  activation = 0;
  readonly inputs: Synapse[] = [];

  connectFrom(source: Neuron, weight?: number): Synapse {
    const synapse = new Synapse(source, weight);
    this.inputs.push(synapse);
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
}

// --- New: one training example ---

/** One input vector and the output we wish the network produced. */
interface Sample {
  readonly inputs: number[];
  readonly target: number[];
}

// --- New: the cost function ---

interface CostFunction {
  readonly name: string;

  /** How wrong is this output? Lower is better, 0 is perfect. */
  evaluate(output: number[], target: number[]): number;

  /**
   * For each output neuron: how does the cost change if that neuron's
   * activation goes up? Positive means "you are too high, come down".
   * This is the only thing step 04 needs from the cost function.
   */
  gradient(output: number[], target: number[]): number[];
}

/** cost = average of (output - target)^2 over the output neurons. */
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
    // d/da of (a - t)^2 is 2 * (a - t).
    // We skip the 1/n from evaluate() on purpose: it is a constant factor
    // that the learning rate absorbs anyway.
    return target.map((t, i) => 2 * (output[i] - t));
  }
}

// --- Network, now aware of its cost function ---

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

  /** Run one sample and score it. */
  evaluate(sample: Sample): number {
    return this.cost.evaluate(this.forward(sample.inputs), sample.target);
  }

  /** The number we will spend the rest of this tutorial trying to shrink. */
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

// --- Measure how bad a random network is ---

const xor: Sample[] = [
  { inputs: [0, 0], target: [0] },
  { inputs: [0, 1], target: [1] },
  { inputs: [1, 0], target: [1] },
  { inputs: [1, 1], target: [0] },
];

const net = new Network([2, 4, 1]);

console.log(`=== Cost on XOR (${net.cost.name}) ===\n`);

for (const sample of xor) {
  const [output] = net.forward(sample.inputs);
  const cost = net.cost.evaluate([output], sample.target);
  const [gradient] = net.cost.gradient([output], sample.target);
  console.log(
    `  [${sample.inputs.join(", ")}]` +
    `  output ${output.toFixed(4)}` +
    `  target ${sample.target[0]}` +
    `  cost ${cost.toFixed(4)}` +
    `  gradient ${gradient >= 0 ? "+" : ""}${gradient.toFixed(4)}` +
    `  (${gradient > 0 ? "too high" : "too low"})`,
  );
}

const average = net.averageCost(xor);
console.log(`\n  Average cost: ${average.toFixed(4)}`);
console.log("\n  For reference:");
console.log("    perfect network        cost = 0.0000");
console.log("    always answers 0.5     cost = 0.2500");
console.log(`    our random network     cost = ${average.toFixed(4)}`);

// --- The cost is a function of the weights ---

console.log("\n=== Cost as a function of ONE weight ===\n");
console.log("  Freeze the whole network, move a single weight, re-measure.");
console.log("  This is the landscape gradient descent walks down -");
console.log("  except the real one has one dimension per weight.\n");

const probe = net.layers[1].neurons[0].inputs[0];
const original = probe.weight;
const readings: { weight: number; cost: number }[] = [];

for (let weight = -8; weight <= 8; weight += 1) {
  probe.weight = weight;
  readings.push({ weight, cost: net.averageCost(xor) });
}
probe.weight = original;

// The spread is small, so stretch it: the shape of the curve is the point,
// not its absolute height.
const worst = Math.max(...readings.map(r => r.cost));
const best = readings.reduce((a, b) => (b.cost < a.cost ? b : a));
const spread = worst - best.cost;

for (const reading of readings) {
  const height = spread === 0 ? 0 : (reading.cost - best.cost) / spread;
  const bar = "█".repeat(Math.round(height * 40));
  const marker = reading === best ? "  <- lowest" : "";
  console.log(`  w = ${reading.weight.toString().padStart(3)}  ${reading.cost.toFixed(4)}  ${bar}${marker}`);
}

console.log(`\n  Best this weight can do alone: ${best.cost.toFixed(4)} (from ${average.toFixed(4)}).`);
console.log("  Barely a dent - and it is still nowhere near 0.");

console.log("\n--- Key insight ---");
console.log("The cost turns 'is this network any good' into a single number,");
console.log("and that number is a smooth function of every weight.");
console.log("Smooth means we can ask: which way is downhill?");
console.log("But the sweep above also shows why one weight at a time is hopeless.");
console.log(`This tiny network already has ${net.parameterCount} parameters, a real one has millions;`);
console.log("we need the slope along ALL of them at once, from one pass.");
console.log("That is backpropagation.");

console.log("\n--- Try it ---");
console.log("Probe a different weight (net.layers[1].neurons[2].inputs[1]).");
console.log("Some weights barely move the cost - the network hardly uses them.");
console.log("Probe the output neuron instead (net.layers[2].neurons[0].inputs[0]):");
console.log("weights near the output usually matter more than weights near the input.");

// Step 02 - The Forward Pass
// ==========================
// Wiring neurons by hand does not scale. Two new objects fix that:
//
//   Layer    - a group of neurons, all connected to the previous layer
//   Network  - an ordered list of layers
//
// Watch what happens to the code once those exist: `Network.forward()`
// is three lines. Not because we hid the work, but because each object
// does its own part. A Neuron knows how to activate itself, a Layer
// knows how to activate its neurons, a Network knows the order of the
// layers. That is the whole point of modelling this with objects.
//
// NEW IN THIS STEP
//   class Layer    -  receive(), activate(), activations
//   class Network  -  forward(), inputLayer, outputLayer
//
// 3Blue1Brown reference: Chapter 1, 5:00-15:00
// https://www.youtube.com/watch?v=aircAruvnKk

function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

// --- Same as step 01 ---

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

// --- New: a layer of neurons ---

/**
 * A layer owns its neurons. When built on top of another layer, every
 * neuron here is connected to every neuron there - a "fully connected"
 * layer, which is the only kind we need for this tutorial.
 */
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

  /** Push raw values in. Only the input layer ever needs this. */
  receive(values: number[]): void {
    this.neurons.forEach((neuron, i) => neuron.receive(values[i]));
  }

  /** Every neuron computes itself from the layer behind it. */
  activate(): void {
    for (const neuron of this.neurons) neuron.activate();
  }
}

// --- New: the network ---

/**
 * A network is layers in order. `new Network([2, 4, 1])` means
 * 2 inputs, one hidden layer of 4, and 1 output.
 */
class Network {
  readonly layers: Layer[];

  constructor(sizes: number[]) {
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

  /** Every layer except the input one - the layers that actually compute. */
  get computedLayers(): Layer[] {
    return this.layers.slice(1);
  }

  /**
   * The forward pass: values enter on the left, activations ripple right.
   * Three lines, because Layer and Neuron already do the real work.
   */
  forward(inputs: number[]): number[] {
    this.inputLayer.receive(inputs);
    for (const layer of this.computedLayers) layer.activate();
    return this.outputLayer.activations;
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

  toString(): string {
    const shape = this.layers.map(layer => layer.size).join(" -> ");
    const count = (n: number) => n.toLocaleString("en-US");
    return `${shape}   (${count(this.parameterCount)} parameters:` +
      ` ${count(this.weightCount)} weights, ${count(this.biasCount)} biases)`;
  }
}

// --- Let's push data through ---

const net = new Network([2, 4, 1]);

console.log("=== Forward pass with random weights ===\n");
console.log(`  Network: ${net}\n`);

const xor = [
  { inputs: [0, 0], target: 0 },
  { inputs: [0, 1], target: 1 },
  { inputs: [1, 0], target: 1 },
  { inputs: [1, 1], target: 0 },
];

for (const sample of xor) {
  const [output] = net.forward(sample.inputs);
  console.log(
    `  Input [${sample.inputs.join(", ")}]` +
    `  ->  ${output.toFixed(4)}` +
    `   (want ${sample.target})`,
  );
}

// --- Look inside ---

console.log("\n=== What the middle layer sees ===\n");
console.log("  The hidden layer is where the network builds its own features.");
console.log("  Right now those features are random noise.\n");

for (const sample of xor) {
  net.forward(sample.inputs);
  const hidden = net.layers[1].activations.map(a => a.toFixed(3)).join(", ");
  console.log(`  Input [${sample.inputs.join(", ")}]  ->  hidden [${hidden}]`);
}

console.log("\n--- Key insight ---");
console.log("The plumbing works: data flows all the way through.");
console.log("But the weights are random, so the four outputs are nearly");
console.log("identical - the network barely reacts to its input, and what");
console.log("little reaction there is means nothing. It knows nothing yet.");
console.log("Note the [0, 0] row: with both inputs at 0 every weighted sum");
console.log("is 0, so every hidden neuron just returns sigmoid(bias).");
console.log("Next: put a number on how wrong all this is.");

console.log("\n--- Try it ---");
console.log("Build a bigger network: new Network([2, 16, 16, 1]).");
console.log("Print it - see how fast the weight count grows with depth.");

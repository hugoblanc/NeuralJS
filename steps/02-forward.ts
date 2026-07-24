// Step 02 - The Forward Pass
// ==========================
// A neural network is layers of neurons connected to each other.
// Data flows from left to right: input -> hidden -> output.
// This is the "forward pass".
//
// Here we build a full network and push data through it.
//
// 3Blue1Brown reference: Chapter 1, 5:00-15:00
// https://www.youtube.com/watch?v=aircAruvnKk

function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

// --- Building blocks ---

interface Synapse {
  weight: number;
  parent: Neuron;
}

class Neuron {
  bias = 1;
  sum = 0;
  activation = 0;
  synapses: Synapse[] = [];

  constructor(parents?: Neuron[]) {
    if (!parents) return;
    this.synapses = parents.map(parent => ({
      weight: Math.random() * 2 - 1,
      parent,
    }));
  }
}

class Layer {
  neurons: Neuron[];

  constructor(size: number, parentLayer?: Layer) {
    const parents = parentLayer?.neurons;
    this.neurons = Array.from({ length: size }, () => new Neuron(parents));
  }
}

class Network {
  layers: Layer[];

  constructor(sizes: number[]) {
    this.layers = [];
    let prev: Layer | undefined;
    for (const size of sizes) {
      const layer = new Layer(size, prev);
      this.layers.push(layer);
      prev = layer;
    }
  }

  // The forward pass: data flows through the network
  forward(inputs: number[]): number[] {
    // Input layer: just store the raw values
    const inputNeurons = this.layers[0].neurons;
    for (let i = 0; i < inputNeurons.length; i++) {
      inputNeurons[i].activation = inputs[i];
    }

    // Each subsequent layer computes: sigmoid(sum(weight * input) + bias)
    for (let l = 1; l < this.layers.length; l++) {
      for (const neuron of this.layers[l].neurons) {
        let sum = 0;
        for (const synapse of neuron.synapses) {
          sum += synapse.parent.activation * synapse.weight;
        }
        neuron.sum = sum + neuron.bias;
        neuron.activation = sigmoid(neuron.sum);
      }
    }

    return this.layers[this.layers.length - 1].neurons.map(n => n.activation);
  }
}

// --- Let's see it in action ---

// A network for XOR: 2 inputs, 4 hidden neurons, 1 output
const net = new Network([2, 4, 1]);

console.log("=== Forward pass with random weights ===\n");
console.log("The network has random weights, so the outputs are meaningless.");
console.log("But the data flows through correctly.\n");

const xorInputs = [[0, 0], [0, 1], [1, 0], [1, 1]];
const xorTargets = [0, 1, 1, 0];

for (let i = 0; i < xorInputs.length; i++) {
  const result = net.forward(xorInputs[i]);
  console.log(
    `  Input: [${xorInputs[i].join(", ")}]` +
    `  Output: ${result[0].toFixed(4)}` +
    `  (want: ${xorTargets[i]})`,
  );
}

console.log("\n--- Key insight ---");
console.log("With random weights, the network outputs ~0.5 for everything.");
console.log("It knows nothing yet. Next step: measuring how wrong it is.");

// Step 04 - Backpropagation
// =========================
// We have a cost. Now we need to reduce it.
// The idea: figure out how much each weight contributed to the error,
// then nudge it in the opposite direction.
//
// This is backpropagation - the error flows backward through the network,
// from output to input, layer by layer.
//
// 3 levers to reduce cost for each neuron:
//   1. Adjust its bias
//   2. Adjust the weights (proportional to parent activations)
//   3. Wish for different parent activations (propagate backward)
//
// 3Blue1Brown reference: Chapter 3
// https://www.youtube.com/watch?v=Ilg3gGewQ5U

function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

function sigmoidDerivative(x: number): number {
  const s = sigmoid(x);
  return s * (1 - s);
}

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
        let sum = 0;
        for (const s of n.synapses) sum += s.parent.activation * s.weight;
        n.sum = sum + n.bias;
        n.activation = sigmoid(n.sum);
      }
    }
    return this.layers[this.layers.length - 1].neurons.map(n => n.activation);
  }

  // THIS IS THE NEW PART: backpropagation
  backpropagate(target: number[], learningRate: number): void {
    const outputNeurons = this.layers[this.layers.length - 1].neurons;

    // Step 1 - How wrong is each output neuron?
    // delta = dCost/dActivation * dActivation/dSum
    for (let i = 0; i < outputNeurons.length; i++) {
      const neuron = outputNeurons[i];
      const costGradient = 2 * (neuron.activation - target[i]);
      neuron.delta = costGradient * sigmoidDerivative(neuron.sum);
    }

    // Step 2 - Propagate the error backward through hidden layers
    // Each neuron's error = how much it contributed to the next layer's error
    for (let l = this.layers.length - 2; l >= 1; l--) {
      const layer = this.layers[l];
      const nextLayer = this.layers[l + 1];
      for (let j = 0; j < layer.neurons.length; j++) {
        const neuron = layer.neurons[j];
        let error = 0;
        for (const nextNeuron of nextLayer.neurons) {
          error += nextNeuron.synapses[j].weight * nextNeuron.delta;
        }
        neuron.delta = error * sigmoidDerivative(neuron.sum);
      }
    }

    // Step 3 - Nudge every weight and bias against the gradient
    for (let l = 1; l < this.layers.length; l++) {
      for (const neuron of this.layers[l].neurons) {
        neuron.bias -= neuron.delta * learningRate;
        for (const synapse of neuron.synapses) {
          synapse.weight -= neuron.delta * synapse.parent.activation * learningRate;
        }
      }
    }
  }
}

// --- Let's watch it learn XOR ---

interface Sample { inputs: number[]; target: number[]; }
const xor: Sample[] = [
  { inputs: [0, 0], target: [0] },
  { inputs: [0, 1], target: [1] },
  { inputs: [1, 0], target: [1] },
  { inputs: [1, 1], target: [0] },
];

const net = new Network([2, 8, 1]);
const learningRate = 0.5;

console.log("=== Learning XOR with backpropagation ===\n");

for (let epoch = 0; epoch <= 10000; epoch++) {
  const sample = xor[epoch % xor.length];
  const output = net.forward(sample.inputs);
  net.backpropagate(sample.target, learningRate);

  if (epoch % 2000 === 0) {
    let totalCost = 0;
    const results: string[] = [];
    for (const s of xor) {
      const r = net.forward(s.inputs);
      totalCost += (r[0] - s.target[0]) ** 2;
      results.push(`[${s.inputs.join(",")}]->${r[0].toFixed(3)}`);
    }
    console.log(
      `  Epoch ${epoch.toString().padStart(5)}` +
      `  Cost: ${(totalCost / 4).toFixed(6)}` +
      `  ${results.join("  ")}`,
    );
  }
}

console.log("\n--- Key insight ---");
console.log("The network learned XOR! Look at how the cost drops.");
console.log("Backpropagation is just the chain rule from calculus,");
console.log("applied layer by layer from output back to input.");
console.log("Each weight gets nudged proportionally to how much it");
console.log("contributed to the error. That's it.");

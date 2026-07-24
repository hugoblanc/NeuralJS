// Step 03 - The Cost Function
// ============================
// How do you tell a network it's doing a bad job?
// You need a number that says "this is how wrong you are."
// That number is the cost (or loss).
//
// cost = average of (expected - actual)^2 for each output
//
// Low cost = good. High cost = bad. Our goal: minimize the cost.
//
// 3Blue1Brown reference: Chapter 2, 0:00-6:00
// https://www.youtube.com/watch?v=IHZwWFHWa-w

function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
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
}

// --- The cost function ---

// Mean Squared Error: average of (target - output)^2
function cost(output: number[], target: number[]): number {
  let sum = 0;
  for (let i = 0; i < target.length; i++) {
    const diff = output[i] - target[i];
    sum += diff * diff;
  }
  return sum / target.length;
}

// --- Let's measure how bad a random network is ---

const net = new Network([2, 4, 1]);

interface Sample { inputs: number[]; target: number[]; }
const xor: Sample[] = [
  { inputs: [0, 0], target: [0] },
  { inputs: [0, 1], target: [1] },
  { inputs: [1, 0], target: [1] },
  { inputs: [1, 1], target: [0] },
];

console.log("=== Measuring cost on XOR ===\n");

let totalCost = 0;
for (const sample of xor) {
  const output = net.forward(sample.inputs);
  const c = cost(output, sample.target);
  totalCost += c;
  console.log(
    `  Input: [${sample.inputs.join(", ")}]` +
    `  Output: ${output[0].toFixed(4)}` +
    `  Target: ${sample.target[0]}` +
    `  Cost: ${c.toFixed(4)}`,
  );
}

const avgCost = totalCost / xor.length;
console.log(`\n  Average cost: ${avgCost.toFixed(4)}`);

console.log("\n  For reference:");
console.log("  - Perfect network: cost = 0.0000");
console.log("  - Always outputs 0.5: cost ~= 0.2500");
console.log(`  - Our random network: cost = ${avgCost.toFixed(4)}`);

console.log("\n--- Key insight ---");
console.log("The cost function gives us a single number to minimize.");
console.log("If we can find weights that make this number small,");
console.log("the network will have learned. But how do we find them?");
console.log("Next step: gradient descent.");

// Step 07 - The Autoencoder
// =========================
// What if the network had to compress data through a bottleneck?
//
// 8 inputs -> 5 neurons -> 3 neurons -> 5 neurons -> 8 outputs
//                          ^^^^^^^^^^^
//                          bottleneck!
//
// The input equals the target: the network must learn to reconstruct
// its own input after squeezing it through just 3 neurons.
//
// 8 possible patterns, 3 neurons -> the network must invent
// a binary encoding to distinguish them. 2^3 = 8. Just enough.
//
// This is the foundation of autoencoders, a key concept in deep learning.

import * as readline from "readline";

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

// --- 8 one-hot patterns: input = target ---

interface Sample { inputs: number[]; target: number[]; }
const samples: Sample[] = [];
for (let i = 0; i < 8; i++) {
  const vec = new Array(8).fill(0);
  vec[i] = 1;
  samples.push({ inputs: [...vec], target: [...vec] });
}

// Architecture: 8 -> 5 -> 3 -> 5 -> 8
const net = new Network([8, 5, 3, 5, 8]);
const lr = 1.0;
const iterations = 100000;

console.log("=== Autoencoder: 8 -> 5 -> [3] -> 5 -> 8 ===");
console.log(`LR: ${lr} | Iterations: ${iterations}\n`);

function bg(r: number, g: number, b: number, text: string): string {
  return `\x1b[48;2;${r};${g};${b}m${text}\x1b[0m`;
}
const history: number[] = [];

for (let i = 0; i < iterations; i++) {
  const sample = samples[i % samples.length];
  const result = net.forward(sample.inputs);

  let precision = 0;
  for (let j = 0; j < sample.target.length; j++)
    precision += 1 - Math.abs(sample.target[j] - result[j]);
  precision /= sample.target.length;

  history.push(precision * 100);
  if (history.length > 10) history.shift();
  const avg = history.reduce((a, b) => a + b) / history.length;

  const barW = 40;
  const filled = Math.round((avg / 100) * barW);
  const bar = "█".repeat(filled) + "░".repeat(barW - filled);
  const line = ` ${avg.toFixed(1)}% ${bar} `;
  const red = Math.round(200 - 150 * (avg / 100));
  const green = Math.round(130 * (avg / 100));

  readline.cursorTo(process.stdout, 0);
  readline.clearLine(process.stdout, 0);
  process.stdout.write(bg(red, green, 0, line));

  net.backpropagate(sample.target, lr);
}

// --- Inspect the bottleneck ---

const bnIndex = Math.floor(net.layers.length / 2);
console.log(`\n\n=== What did the bottleneck learn? (layer ${bnIndex}, 3 neurons) ===\n`);
console.log("  The network must encode 8 patterns using only 3 neurons.");
console.log("  Each neuron acts like a bit (close to 0 or 1).");
console.log("  2^3 = 8 -> just enough to distinguish all patterns.\n");

for (let i = 0; i < samples.length; i++) {
  const sample = samples[i];
  net.forward(sample.inputs);
  const bottleneck = net.layers[bnIndex].neurons.map(n => n.activation);
  const rounded = bottleneck.map(v => Math.round(v));
  const output = net.layers[net.layers.length - 1].neurons.map(n => n.activation);
  const maxOut = Math.max(...output);
  const reconstructed = output.map(v => v > 0.5 ? 1 : 0);

  console.log(
    `  Pattern ${i}: [${sample.inputs.join("")}]` +
    `  -> bottleneck [${bottleneck.map(v => v.toFixed(2)).join(", ")}]` +
    ` ~ [${rounded.join("")}]` +
    `  -> output [${reconstructed.join("")}]`,
  );
}

console.log("\n--- Key insight ---");
console.log("The network invented its own binary code!");
console.log("Nobody told it about binary - it discovered that 3 neurons");
console.log("with values near 0 or 1 can represent 8 distinct patterns.");
console.log("This is the core idea behind autoencoders: learn a compressed");
console.log("representation of the data, then reconstruct from it.");

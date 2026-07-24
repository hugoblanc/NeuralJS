// Step 05 - The Training Loop
// ============================
// Now we put it all together: forward pass, measure cost,
// backpropagate, repeat. With a visual progress bar.
//
// We also add the learning rate, which controls how big
// each step is. Too big = overshooting. Too small = slow.
//
// Try it: change the learning rate and see what happens.
//   npx tsx steps/05-training.ts 0.1    (slow but stable)
//   npx tsx steps/05-training.ts 2.0    (fast but risky)
//   npx tsx steps/05-training.ts 10.0   (chaos)
//
// 3Blue1Brown reference: Chapter 2, 6:00-15:00
// https://www.youtube.com/watch?v=IHZwWFHWa-w

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
    for (let i = 0; i < out.length; i++) {
      out[i].delta = 2 * (out[i].activation - target[i]) * sigmoidDeriv(out[i].sum);
    }
    for (let l = this.layers.length - 2; l >= 1; l--) {
      for (let j = 0; j < this.layers[l].neurons.length; j++) {
        const n = this.layers[l].neurons[j];
        let err = 0;
        for (const nn of this.layers[l + 1].neurons) err += nn.synapses[j].weight * nn.delta;
        n.delta = err * sigmoidDeriv(n.sum);
      }
    }
    for (let l = 1; l < this.layers.length; l++) {
      for (const n of this.layers[l].neurons) {
        n.bias -= n.delta * lr;
        for (const syn of n.synapses) syn.weight -= n.delta * syn.parent.activation * lr;
      }
    }
  }
}

// --- Training loop with live display ---

interface Sample { inputs: number[]; target: number[]; }

const datasets: Record<string, { desc: string; samples: Sample[]; layers: number[] }> = {
  xor: {
    desc: "XOR gate",
    layers: [2, 8, 1],
    samples: [
      { inputs: [0, 0], target: [0] },
      { inputs: [0, 1], target: [1] },
      { inputs: [1, 0], target: [1] },
      { inputs: [1, 1], target: [0] },
    ],
  },
  parity: {
    desc: "Parity check (4 bits)",
    layers: [4, 8, 1],
    samples: [
      { inputs: [0,0,0,0], target: [0] }, { inputs: [1,0,0,0], target: [1] },
      { inputs: [0,1,0,0], target: [1] }, { inputs: [0,0,1,0], target: [1] },
      { inputs: [0,0,0,1], target: [1] }, { inputs: [1,1,0,0], target: [0] },
      { inputs: [1,0,1,0], target: [0] }, { inputs: [1,0,0,1], target: [0] },
      { inputs: [0,1,1,0], target: [0] }, { inputs: [0,1,0,1], target: [0] },
      { inputs: [0,0,1,1], target: [0] }, { inputs: [1,1,1,0], target: [1] },
      { inputs: [1,1,0,1], target: [1] }, { inputs: [1,0,1,1], target: [1] },
      { inputs: [0,1,1,1], target: [1] }, { inputs: [1,1,1,1], target: [0] },
    ],
  },
};

const datasetKey = process.argv[2] || "xor";
const lr = parseFloat(process.argv[3] || "0.5");
const maxIter = parseInt(process.argv[4] || "30000", 10);
const ds = datasets[datasetKey] || datasets.xor;

console.log(`=== Training: ${ds.desc} | LR: ${lr} | Iterations: ${maxIter} ===\n`);

const net = new Network(ds.layers);
const history: number[] = [];

function bg(r: number, g: number, b: number, text: string): string {
  return `\x1b[48;2;${r};${g};${b}m${text}\x1b[0m`;
}

for (let i = 0; i < maxIter; i++) {
  const sample = ds.samples[i % ds.samples.length];
  const result = net.forward(sample.inputs);

  let precision = 0;
  for (let j = 0; j < sample.target.length; j++) {
    precision += 1 - Math.abs(sample.target[j] - result[j]);
  }
  precision /= sample.target.length;

  history.push(precision * 100);
  if (history.length > 10) history.shift();
  const avg = history.reduce((a, b) => a + b) / history.length;

  const barW = 40;
  const filled = Math.round((avg / 100) * barW);
  const bar = "█".repeat(filled) + "░".repeat(barW - filled);
  const t = sample.target.map(v => v.toFixed(0)).join(" ");
  const r2 = result.map(v => v.toFixed(3)).join(" ");
  const line = ` ${avg.toFixed(1)}% ${bar}  T:[${t}]  R:[${r2}] `;
  const red = Math.round(200 - 150 * (avg / 100));
  const green = Math.round(130 * (avg / 100));

  readline.cursorTo(process.stdout, 0);
  readline.clearLine(process.stdout, 0);
  process.stdout.write(bg(red, green, 0, line));

  net.backpropagate(sample.target, lr);
}

console.log("\n\n=== Final results ===\n");
for (const s of ds.samples) {
  const r = net.forward(s.inputs);
  const ok = Math.abs(r[0] - s.target[0]) < 0.1 ? "ok" : "MISS";
  console.log(`  [${s.inputs.join(",")}] -> ${r[0].toFixed(3)} (want ${s.target[0]})  ${ok}`);
}

console.log("\n--- Try it ---");
console.log("npx tsx steps/05-training.ts xor 0.5       (default)");
console.log("npx tsx steps/05-training.ts xor 0.05      (slow learning rate)");
console.log("npx tsx steps/05-training.ts xor 5.0       (too fast - does it still work?)");
console.log("npx tsx steps/05-training.ts parity 0.5    (harder problem)");

// Step 01 - The Neuron
// ====================
// A neuron is just a number between 0 and 1.
// That's it. No magic.
//
// In this step we build the smallest piece of a neural network:
// a single neuron with weighted connections to its inputs.
//
// 3Blue1Brown reference: Chapter 1, 0:00-5:00
// https://www.youtube.com/watch?v=aircAruvnKk

// A synapse connects two neurons. It has a weight that determines
// how strongly the input influences the output.
interface Synapse {
  weight: number;
  input: number;
}

// A neuron computes a weighted sum of its inputs, adds a bias,
// then squishes the result between 0 and 1 with the sigmoid function.
function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

function neuronOutput(synapses: Synapse[], bias: number): number {
  let sum = 0;
  for (const s of synapses) {
    sum += s.input * s.weight;
  }
  return sigmoid(sum + bias);
}

// --- Let's play with it ---

// A neuron with 3 inputs
const synapses: Synapse[] = [
  { weight: 0.5, input: 1.0 },
  { weight: -0.3, input: 0.8 },
  { weight: 0.9, input: 0.2 },
];
const bias = -0.1;

const output = neuronOutput(synapses, bias);
console.log("=== A single neuron ===\n");
console.log("Inputs:  ", synapses.map(s => s.input));
console.log("Weights: ", synapses.map(s => s.weight));
console.log("Bias:    ", bias);
console.log("Output:  ", output.toFixed(4));

// Try it: what happens when you change the weights or bias?

console.log("\n=== What does sigmoid do? ===\n");
const testValues = [-5, -2, -1, -0.5, 0, 0.5, 1, 2, 5];
for (const x of testValues) {
  const y = sigmoid(x);
  const bar = "█".repeat(Math.round(y * 40));
  console.log(`  sigmoid(${x.toString().padStart(4)}) = ${y.toFixed(4)}  ${bar}`);
}

console.log("\n--- Key insight ---");
console.log("Sigmoid squishes any number into the range [0, 1].");
console.log("Large negative -> ~0, large positive -> ~1, zero -> 0.5");
console.log("This is what makes a neuron's output an 'activation'.");

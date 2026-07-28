/**
 * Gradient check.
 *
 * Backpropagation claims to compute dCost/dWeight for every weight in one
 * backward pass. There is an obvious, slow way to check that claim: nudge
 * a single weight up and down by a tiny amount, measure how the cost moves,
 * and divide. That is the definition of a derivative.
 *
 * If backpropagation is implemented correctly the two must agree to within
 * floating-point noise. If it is subtly wrong - a missing term, a sign flip,
 * a derivative taken at the wrong point - the network will often still train
 * and still look convincing, but this check will disagree loudly.
 *
 * Run it with:  npm run gradcheck
 */
import { Network } from './network';

const net = new Network([3, 4, 3, 2]);
const inputs = [0.3, 0.7, 0.1];
const target = [1, 0];

/**
 * MeanSquaredError.gradient() drops the 1/n that evaluate() applies, so the
 * loss the analytic deltas actually correspond to is the SUM of squares.
 * Comparing against the mean would fail by exactly a factor of n.
 */
function loss(): number {
  const output = net.forward(inputs);
  let total = 0;
  for (let i = 0; i < target.length; i++) {
    total += (output[i] - target[i]) ** 2;
  }
  return total;
}

// One forward and one backward pass fills in every neuron's delta.
// A learning rate of 0 means nothing is modified.
net.forward(inputs);
net.backpropagate(target, 0);

const h = 1e-5;
let worstWeightError = 0;
let worstBiasError = 0;
let checked = 0;

for (const layer of net.layers.slice(1)) {
  for (const neuron of layer.neurons) {
    // dCost/dBias is just the delta.
    {
      const analytic = neuron.delta;
      const original = neuron.bias;
      neuron.bias = original + h;
      const up = loss();
      neuron.bias = original - h;
      const down = loss();
      neuron.bias = original;

      worstBiasError = Math.max(worstBiasError, Math.abs(analytic - (up - down) / (2 * h)));
      checked++;
    }

    // dCost/dWeight is the delta scaled by whatever the source sent.
    for (const synapse of neuron.inputs) {
      net.forward(inputs); // the bias probes above left stale activations
      const analytic = neuron.delta * synapse.source.activation;
      const original = synapse.weight;
      synapse.weight = original + h;
      const up = loss();
      synapse.weight = original - h;
      const down = loss();
      synapse.weight = original;

      worstWeightError = Math.max(worstWeightError, Math.abs(analytic - (up - down) / (2 * h)));
      checked++;
    }
  }
}

console.log(`Network             ${net.layers.map(l => l.size).join(' -> ')}`);
console.log(`Parameters checked  ${checked}`);
console.log(`Worst weight error  ${worstWeightError.toExponential(3)}`);
console.log(`Worst bias error    ${worstBiasError.toExponential(3)}`);

const TOLERANCE = 1e-6;
if (worstWeightError < TOLERANCE && worstBiasError < TOLERANCE) {
  console.log('\nPASS - backpropagation agrees with the numerical gradients.');
} else {
  console.error(`\nFAIL - disagreement above the ${TOLERANCE} tolerance.`);
  process.exit(1);
}

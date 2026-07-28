// Step 01 - The Neuron
// ====================
// A neuron is just a number between 0 and 1.
// That's it. No magic.
//
// But a neuron never lives alone. It is wired to other neurons by
// synapses, and each synapse carries a weight. So even the smallest
// useful model already has two things in it:
//
//   Synapse  - a weighted connection, knows which neuron it reads from
//   Neuron   - holds a value, and knows how to compute it from its synapses
//
// Both have state (a weight, an activation) and behaviour (compute
// yourself), so both are classes. Throughout this tutorial the object
// graph IS the network: no matrices, no tensors. If you can picture the
// objects pointing at each other, you understand the algorithm.
//
// NEW IN THIS STEP
//   class Synapse
//   class Neuron  -  receive(), connectFrom(), activate()
//
// 3Blue1Brown reference: Chapter 1, 0:00-5:00
// https://www.youtube.com/watch?v=aircAruvnKk

// Sigmoid squishes any real number into [0, 1].
// It is what turns a raw weighted sum into an "activation".
function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

/**
 * A weighted connection between two neurons.
 * The weight is the thing a neural network actually learns -
 * everything else in this file is just plumbing around it.
 */
class Synapse {
  weight: number;

  constructor(
    readonly source: Neuron,
    weight: number = Math.random() * 2 - 1,
  ) {
    this.weight = weight;
  }

  /** What this connection contributes to the neuron it feeds. */
  get contribution(): number {
    return this.source.activation * this.weight;
  }
}

/**
 * A neuron holds exactly one number: its activation, between 0 and 1.
 * It computes that number from the neurons it is connected to.
 */
class Neuron {
  /**
   * Shifts the sum before the sigmoid, so it sets how easily this neuron
   * fires at all. A large negative bias means the inputs have to work hard
   * to switch it on; a large positive one means it is on unless argued out of it.
   */
  bias = 1;

  /** The raw weighted sum + bias, before sigmoid. We will need it in step 04. */
  sum = 0;

  /** This neuron's output, in [0, 1]. */
  activation = 0;

  /** Incoming connections. An input neuron has none. */
  readonly inputs: Synapse[] = [];

  /** Wire this neuron so that it reads from `source`. */
  connectFrom(source: Neuron, weight?: number): Synapse {
    const synapse = new Synapse(source, weight);
    this.inputs.push(synapse);
    return synapse;
  }

  /**
   * Input neurons don't compute anything - they just hold a raw value.
   * They are the only neurons that get their activation from outside.
   */
  receive(value: number): void {
    this.sum = value;
    this.activation = value;
  }

  /**
   * The one formula that matters:
   *   activation = sigmoid( sum(weight * source.activation) + bias )
   */
  activate(): number {
    let total = 0;
    for (const synapse of this.inputs) total += synapse.contribution;
    this.sum = total + this.bias;
    this.activation = sigmoid(this.sum);
    return this.activation;
  }
}

// --- Let's build one by hand ---

console.log("=== A single neuron ===\n");

// Three input neurons, holding raw values.
const inputs = [new Neuron(), new Neuron(), new Neuron()];
inputs[0].receive(1.0);
inputs[1].receive(0.8);
inputs[2].receive(0.2);

// One neuron that reads from all three, with weights we choose ourselves.
const neuron = new Neuron();
neuron.bias = -0.1;
neuron.connectFrom(inputs[0], 0.5);
neuron.connectFrom(inputs[1], -0.3);
neuron.connectFrom(inputs[2], 0.9);

neuron.activate();

console.log("  Each synapse contributes source.activation * weight:\n");
for (const synapse of neuron.inputs) {
  console.log(
    `    ${synapse.source.activation.toFixed(1)} * ${synapse.weight.toString().padStart(4)}` +
    ` = ${synapse.contribution.toFixed(3).padStart(6)}`,
  );
}
console.log(`    bias                = ${neuron.bias.toFixed(3).padStart(6)}`);
console.log(`    ${"-".repeat(28)}`);
console.log(`    sum                 = ${neuron.sum.toFixed(3).padStart(6)}`);
console.log(`    sigmoid(sum)        = ${neuron.activation.toFixed(3).padStart(6)}  <- the activation\n`);

// --- Why sigmoid? ---

console.log("=== What sigmoid does ===\n");
for (const x of [-5, -2, -1, -0.5, 0, 0.5, 1, 2, 5]) {
  const y = sigmoid(x);
  console.log(`  sigmoid(${x.toString().padStart(4)}) = ${y.toFixed(4)}  ${"█".repeat(Math.round(y * 40))}`);
}

console.log("\n--- Key insight ---");
console.log("Sigmoid squishes any number into [0, 1]:");
console.log("very negative -> ~0, very positive -> ~1, zero -> 0.5.");
console.log("That bound is what lets us read a neuron's output as");
console.log("'how strongly is this feature present', on a fixed scale.");

console.log("\n--- Try it ---");
console.log("Change the first weight to 5.0, or the bias to -6, and re-run.");
console.log("Either one saturates the neuron: the sum lands far from 0, where");
console.log("sigmoid is almost flat. Now moving a weight by a whole 1.0 shifts");
console.log("the output by less than 0.01, against about 0.2 at the default bias.");
console.log("A saturated neuron barely responds to anything - and in step 04 you");
console.log("will see that it barely learns anything either, for the same reason.");

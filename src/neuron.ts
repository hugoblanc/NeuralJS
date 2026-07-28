import { ActivationFn } from './activation';

/**
 * A weighted connection between two neurons. It knows both ends: the
 * forward pass walks it from source to target, backpropagation walks it
 * the other way.
 */
export class Synapse {
  weight: number;

  constructor(
    readonly source: Neuron,
    readonly target: Neuron,
    weight: number = Math.random() * 2 - 1,
  ) {
    this.weight = weight;
  }

  /** Forward: what this connection sends. */
  get contribution(): number {
    return this.source.activation * this.weight;
  }

  /** Backward: how much of the target's blame belongs to the source. */
  get blame(): number {
    return this.weight * this.target.delta;
  }
}

export class Neuron {
  bias = 1;
  sum = 0;
  activation = 0;

  /** dCost/dSum for this neuron, filled in during backpropagation. */
  delta = 0;

  readonly inputs: Synapse[] = [];
  readonly outputs: Synapse[] = [];

  connectFrom(source: Neuron, weight?: number): Synapse {
    const synapse = new Synapse(source, this, weight);
    this.inputs.push(synapse);
    source.outputs.push(synapse);
    return synapse;
  }

  /** Input neurons hold a raw value: no weighted sum, no activation function. */
  receive(value: number): void {
    this.sum = value;
    this.activation = value;
  }

  activate(activation: ActivationFn): number {
    let total = 0;
    for (const synapse of this.inputs) total += synapse.contribution;
    this.sum = total + this.bias;
    this.activation = activation.apply(this.sum);
    return this.activation;
  }

  /** Output neuron: the cost function says directly how wrong we are. */
  computeOutputDelta(costGradient: number, activation: ActivationFn): void {
    this.delta = costGradient * activation.derivative(this.sum);
  }

  /** Hidden neuron: collect the blame from the neurons we feed. */
  computeHiddenDelta(activation: ActivationFn): void {
    let error = 0;
    for (const synapse of this.outputs) error += synapse.blame;
    this.delta = error * activation.derivative(this.sum);
  }

  applyGradient(learningRate: number): void {
    this.bias -= this.delta * learningRate;
    for (const synapse of this.inputs) {
      synapse.weight -= this.delta * synapse.source.activation * learningRate;
    }
  }
}

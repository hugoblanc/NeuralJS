import { Layer } from './layer';
import { ActivationFn, sigmoid } from './activation';
import { CostFunction, MeanSquaredError } from './cost';

export class Network {
  readonly layers: Layer[];

  constructor(
    sizes: number[],
    private readonly activation: ActivationFn = sigmoid,
    readonly cost: CostFunction = new MeanSquaredError(),
  ) {
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

  /** Every layer except the input one - the ones that actually compute. */
  get computedLayers(): Layer[] {
    return this.layers.slice(1);
  }

  get bottleneckLayer(): Layer {
    return this.layers[Math.floor(this.layers.length / 2)];
  }

  get parameterCount(): number {
    return this.computedLayers.reduce(
      (total, layer) =>
        total + layer.size + layer.neurons.reduce((n, neuron) => n + neuron.inputs.length, 0),
      0,
    );
  }

  /** Values enter on the left, activations ripple right. */
  forward(inputs: number[]): number[] {
    this.inputLayer.receive(inputs);
    for (const layer of this.computedLayers) layer.activate(this.activation);
    return this.outputLayer.activations;
  }

  /**
   * The same shape as forward(), reversed. Call it right after forward():
   * it reads the sums and activations that pass left behind.
   */
  backpropagate(target: number[], learningRate: number): void {
    // 1. The output layer is the only one the cost function can talk to.
    this.outputLayer.computeOutputDeltas(
      this.cost.gradient(this.outputLayer.activations, target),
      this.activation,
    );

    // 2. Blame travels right to left. Order matters: a layer can only
    //    compute its deltas once the layer it feeds already has its own.
    for (let i = this.layers.length - 2; i >= 1; i--) {
      this.layers[i].computeHiddenDeltas(this.activation);
    }

    // 3. Now every neuron knows its delta, so every neuron can fix itself.
    for (const layer of this.computedLayers) layer.applyGradient(learningRate);
  }

  toString(): string {
    return this.layers.map((layer, i) => {
      const neurons = layer.neurons.map((neuron, j) => {
        const weights = neuron.inputs.map(s => s.weight.toFixed(4)).join(', ');
        return `  Neuron ${j}: act=${neuron.activation.toFixed(4)} bias=${neuron.bias.toFixed(4)} w=[${weights}]`;
      }).join('\n');
      return `Layer ${i} (${layer.size} neurons):\n${neurons}`;
    }).join('\n\n');
  }
}

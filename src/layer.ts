import { Neuron } from './neuron';
import { ActivationFn } from './activation';

export class Layer {
  readonly neurons: Neuron[];

  constructor(size: number, previous?: Layer) {
    this.neurons = Array.from({ length: size }, () => new Neuron());
    if (previous) this.fullyConnect(previous);
  }

  private fullyConnect(previous: Layer): void {
    for (const neuron of this.neurons) {
      for (const source of previous.neurons) neuron.connectFrom(source);
    }
  }

  get size(): number {
    return this.neurons.length;
  }

  get activations(): number[] {
    return this.neurons.map(neuron => neuron.activation);
  }

  receive(values: number[]): void {
    this.neurons.forEach((neuron, i) => neuron.receive(values[i]));
  }

  activate(activation: ActivationFn): void {
    for (const neuron of this.neurons) neuron.activate(activation);
  }

  computeOutputDeltas(costGradient: number[], activation: ActivationFn): void {
    this.neurons.forEach((neuron, i) => neuron.computeOutputDelta(costGradient[i], activation));
  }

  computeHiddenDeltas(activation: ActivationFn): void {
    for (const neuron of this.neurons) neuron.computeHiddenDelta(activation);
  }

  applyGradient(learningRate: number): void {
    for (const neuron of this.neurons) neuron.applyGradient(learningRate);
  }
}

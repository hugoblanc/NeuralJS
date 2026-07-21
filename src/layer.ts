import { Neuron } from './neuron';

export class Layer {
  neurons: Neuron[];

  constructor(size: number, parentLayer?: Layer) {
    const parents = parentLayer?.neurons;
    this.neurons = Array.from({ length: size }, () => new Neuron(parents));
  }
}

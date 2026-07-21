import { Layer } from './layer';
import { ActivationFn, sigmoid } from './activation';

export class Network {
  layers: Layer[];
  private activation: ActivationFn;

  constructor(sizes: number[], activation: ActivationFn = sigmoid) {
    this.activation = activation;
    this.layers = [];
    let prev: Layer | undefined;
    for (const size of sizes) {
      const layer = new Layer(size, prev);
      this.layers.push(layer);
      prev = layer;
    }
  }

  forward(inputs: number[]): number[] {
    // Input layer: store raw values, no activation function
    const inputNeurons = this.layers[0].neurons;
    for (let i = 0; i < inputNeurons.length; i++) {
      inputNeurons[i].activation = inputs[i];
      inputNeurons[i].sum = inputs[i];
    }

    // Hidden & output layers: weighted sum + bias -> activation function
    for (let l = 1; l < this.layers.length; l++) {
      for (const neuron of this.layers[l].neurons) {
        let sum = 0;
        for (const synapse of neuron.synapses) {
          sum += synapse.parent.activation * synapse.weight;
        }
        neuron.sum = sum + neuron.bias;
        neuron.activation = this.activation.apply(neuron.sum);
      }
    }

    return this.outputLayer.neurons.map(n => n.activation);
  }

  // MSE cost = mean((activation - target)^2)
  // dCost/dActivation = 2 * (activation - target)
  backpropagate(target: number[], learningRate: number): void {
    // Step 1 - Output layer deltas
    // delta = dCost/dActivation * dActivation/dSum
    const outputNeurons = this.outputLayer.neurons;
    for (let i = 0; i < outputNeurons.length; i++) {
      const neuron = outputNeurons[i];
      const costGradient = 2 * (neuron.activation - target[i]);
      neuron.delta = costGradient * this.activation.derivative(neuron.sum);
    }

    // Step 2 - Hidden layer deltas (right to left)
    // Each neuron's error = sum of (weight * delta) from the next layer
    for (let l = this.layers.length - 2; l >= 1; l--) {
      const layer = this.layers[l];
      const nextLayer = this.layers[l + 1];
      for (let j = 0; j < layer.neurons.length; j++) {
        const neuron = layer.neurons[j];
        let error = 0;
        for (const nextNeuron of nextLayer.neurons) {
          error += nextNeuron.synapses[j].weight * nextNeuron.delta;
        }
        neuron.delta = error * this.activation.derivative(neuron.sum);
      }
    }

    // Step 3 - Update weights and biases against the gradient
    for (let l = 1; l < this.layers.length; l++) {
      for (const neuron of this.layers[l].neurons) {
        neuron.bias -= neuron.delta * learningRate;
        for (const synapse of neuron.synapses) {
          synapse.weight -= neuron.delta * synapse.parent.activation * learningRate;
        }
      }
    }
  }

  get outputLayer(): Layer {
    return this.layers[this.layers.length - 1];
  }

  get bottleneckLayer(): Layer {
    return this.layers[Math.floor(this.layers.length / 2)];
  }

  toString(): string {
    return this.layers.map((layer, i) => {
      const neurons = layer.neurons.map((n, j) => {
        const weights = n.synapses.map(s => s.weight.toFixed(4)).join(', ');
        return `  Neuron ${j}: act=${n.activation.toFixed(4)} bias=${n.bias.toFixed(4)} w=[${weights}]`;
      }).join('\n');
      return `Layer ${i} (${layer.neurons.length} neurons):\n${neurons}`;
    }).join('\n\n');
  }
}

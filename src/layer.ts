import { Neuron } from './neuron';
import { doNTimes } from './util';
export class Layer {

    neurons: Neuron[];

    constructor(nbNeurons: number, prevLayer?: Layer) {
        this.neurons = doNTimes(nbNeurons, () => new Neuron(prevLayer))
    }

}
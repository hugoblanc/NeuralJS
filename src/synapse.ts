import { Neuron } from './neuron';
import { decimal } from './util';
export class Synapse {

    public parentNeuron: Neuron;
    public weight: number;

    constructor(parentNeuron: Neuron) {
        this.parentNeuron = parentNeuron;
        this.weight = (Math.random() - 1) * 2;
    }

    toString(): string {
        return decimal(this.weight, 4);
    }
}
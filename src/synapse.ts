import { Neuron } from './neuron';
export class Synapse {

    public parentNeuron: Neuron;
    public weight: number;

    constructor(parentNeuron: Neuron) {

        this.parentNeuron = parentNeuron;
        this.weight = (Math.random() - 1) * 2;
    }
}
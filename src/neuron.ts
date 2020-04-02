import { Synapse } from './synapse';
import { Layer } from './layer';

export class Neuron {

    public bias = 1;
    public sum = 0;
    public activation = 0;

    public synapses!: Synapse[];

    constructor(parentLayer?: Layer) {
        if (!parentLayer) {
            return;
        }

        this.synapses = [];
        for (const parentNeuron of parentLayer.neurons) {
            this.synapses.push(new Synapse(parentNeuron));
        }
    }


    toString(): string {

    }
}
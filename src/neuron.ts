import { Synapse } from './synapse';

export class Neuron {

    public bias = 1;
    public sum = 0;

    public synapses!: Synapse[];

    public activation = 0;

    constructor(parentLayer?: Neuron[]) {
        if (!parentLayer) {
            return;
        }

        this.synapses = [];
        for (const parentNeuron of parentLayer) {
            this.synapses.push(new Synapse(parentNeuron));
        }
    }


    toString(): string {

    }
}
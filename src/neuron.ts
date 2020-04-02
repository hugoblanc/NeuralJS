import { Synapse } from './synapse';
import { Layer } from './layer';
import { decimal } from './util';

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
        let str = '';
        str += `Acc:${this.dActivation}  Sum: ${this.dSum}  Bias:${this.dBias}  `;

        if(!this.synapses){
            return str;
        }

        str += this.synapses.map((s, i) => `w${i}:${s.toString()}`).join('  ');
        return str;
    }

    private get dActivation() { return decimal(this.activation, 4) }
    private get dSum() { return decimal(this.sum, 4); }
    private get dBias() { return decimal(this.bias, 4); }
}
export interface Synapse {
  weight: number;
  parent: Neuron;
}

export class Neuron {
  bias = 1;
  sum = 0;
  activation = 0;
  delta = 0;
  synapses: Synapse[] = [];

  constructor(parents?: Neuron[]) {
    if (!parents) return;
    this.synapses = parents.map(parent => ({
      weight: Math.random() * 2 - 1,
      parent,
    }));
  }
}

var util = require('util')
const datas = [
    { values: [1], target: 1 },
    { values: [0], target: 0 },
    // { values: [0, 1, 0, 1, 1, 0, 0, 0, 0], target: 1 },
    // { values: [0, 0, 0, 0, 1, 1, 1, 1, 1], target: 0 },
    // { values: [1, 0, 1, 1, 1, 0, 0, 0, 0], target: 1 },
    // { values: [0, 0, 0, 0, 1, 1, 0, 1, 1], target: 0 },
];
const velocity = 0.1;


let globActivateCounter = 0;
let globRecursiveCounter = 0;
let globCostCalcul = 0;

let localRecursive = 0;

function sigmoid(x) {
    return 1 / (1 + Math.pow(Math.E, -x))
}
function derivedSigmoid(x) {
    return sigmoid(x) * (1 - sigmoid(x));
}

class Neuron {
    constructor(nbParents, parentLayer = []) {
        // Init synapses
        this.synapses = [];
        this.sum = 0;
        this.activation = 0;
        this.childSynapses = [];
        for (let i = 0; i < nbParents; i++) {
            const parentNeuron = parentLayer[i] ? parentLayer[i] : null;
            this.synapses.push({ weight: i===0 ? 1 : (Math.random() - 1) * 2, parentNeuron });
            if (parentNeuron) {
                parentNeuron.childSynapses.push(this);
            }
        }
    }

    activate(inputs) {
        let total = 0;
        for (let i = 0; i < inputs.length; i++) {
            const weight = this.synapses[i] ? this.synapses[i].weight : 1;
            const wN = inputs[i] * weight;
            total += wN;
        }

        this.sum = total;
        this.activation = sigmoid(total);
        globActivateCounter++;
        return this.activation;
    }
}


class Network {
    constructor(...nb) {
        if (nb.length <= 1) {
            throw new Error('You motherfucker are forgeting some layers')
        }
        this.layers = [];
        for (let i = 0; i < nb.length; i++) {
            const prevLayer = nb[(i > 0 ? i - 1 : 0)];
            const nextLayer = nb[i];
            // console.log(prevLayer,' and ', nextLayer);

            this.layers.push(this.createLayer(prevLayer, nextLayer))

        }
        this.addFinalLayer(this.layers[this.layers.length - 1]);

    }

    createLayer(nbParent, nbNeurons) {
        let layer = [];
        for (let i = 0; i < nbNeurons; i++) {
            const prevLayer = this.layers.length > 0 ? this.layers[this.layers.length - 1] : undefined;
            layer.push(new Neuron(nbParent, prevLayer));
        }
        return layer;
    }

    addFinalLayer(prevLayer) {
        this.layers.push(this.createLayer(prevLayer.length, 1))
    }

    activate(inputs, layerIndex = 0) {

        if (layerIndex === this.layers.length) {
            return inputs;
        }

        const accFuncResults = [];
        for (const neuron of this.layers[layerIndex]) {
            accFuncResults.push(neuron.activate(inputs));
        }
        return this.activate(accFuncResults, ++layerIndex);
    }


    backPropagationBis(target) {
        const lastLayer = this.layers[this.layers.length - 1];

        for (const finalNeuron of lastLayer) {
            // console.log(finalNeuron);
            const dC = 2 * (finalNeuron.activation - target);
            for (const synapse of finalNeuron.synapses) {
                const dCostPZ = this.derivedCostPartialZ(dC, synapse);
                localRecursive = 0;
                // console.log(++localRecursive);

                this.recPropagation(dC, finalNeuron, dCostPZ, synapse);
                localRecursive = 0;
            }
        }
    }


    recPropagation(dC, finalNeuron, dCostPZ, ...synapses) {
        globRecursiveCounter++;
        const youngestSynapse = synapses[synapses.length - 1];


        if (youngestSynapse.parentNeuron == null) return;
        const parentNeuron = youngestSynapse.parentNeuron;
        let oldWeight = youngestSynapse.weight;
        youngestSynapse.weight = youngestSynapse.weight + (dCostPZ * parentNeuron.activation) * (-velocity);
        // console.log(`Chaine size: ${synapses.length} last${oldWeight} new ${youngestSynapse.weight}  diff ${youngestSynapse.weight - oldWeight}`);

        if (parentNeuron.synapses == null) return;
        for (let i = 0; i < parentNeuron.synapses.length; i++) {
            const pSynapse = parentNeuron.synapses[i];
            if (pSynapse.parentNeuron !== null) {
                dCostPZ = this.derivedCostPartialZ(dC, pSynapse) * youngestSynapse.weight * dCostPZ;
                this.recPropagation(dC, finalNeuron, dCostPZ, ...synapses, pSynapse)
            }
        }
    }

    derivedCostPartialZ(dC, synapse) {
        globCostCalcul++;
        const dS = derivedSigmoid(synapse.parentNeuron.sum);
        // const prevActivation = synapse.parentNeuron.activation;
        return dC * dS;
    }

}


const network = new Network(datas[0].values.length, 2, 5, 10);
// console.log(network);

let result;
for (let i = 0; i < 30000; i++) {
    result = network.activate(datas[i % datas.length].values);
    // console.log(util.inspect(network, {depth: 10}));
    console.log(result, 'Predict => ', (result > 0.5) ? 1 : 0,  ' should be => ' + datas[i % datas.length].target);
    network.backPropagationBis(datas[i % datas.length].target)
    // console.log(util.inspect(network, {depth: 10}));
}




console.log("Neuron activation" + globActivateCounter);
console.log("BackProp recursive call" + globRecursiveCounter);
console.log("Cost derived calculation" + globCostCalcul);

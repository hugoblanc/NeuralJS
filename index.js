const datas = [
    {values: [1, 0, 1, 0, 1, 0, 0, 0, 0], target:1},
    {values: [0, 0, 0, 0, 0, 1, 1, 1, 1], target:0},
    {values: [0, 1, 0, 1, 1, 0, 0, 0, 0], target:1},
    {values: [0, 0, 0, 0, 1, 1, 1, 1, 1], target:0},
    {values: [1, 0, 1, 1, 1, 0, 0, 0, 0], target:1},
    {values: [0, 0, 0, 0, 1, 1, 0, 1, 1], target:0},
];
const velocity = 0.00000001;

function sigmoid(x) {
    return 1 / (1 + Math.pow(Math.E, -x))
}

function derivedSigmoid(x) {
    return sigmoid(x) * (1 - sigmoid(x));
}

class Neuron {
    constructor(nbParents) {
        // Init weights
        this.weights = [];
        this.sum = 0;
        this.activation = 0;
        for (let i = 0; i < nbParents; i++) {
            this.weights.push((Math.random() - 0.5) * 2);
        }
    }

    activate(inputs) {
        let total = 0;
        for (let i = 0; i < inputs.length; i++) {
            const wN = inputs[i] * this.weights[i];
            total += wN;
        }

        this.sum = total;
        this.activation = sigmoid(total);
        return this.activation;
    }
}


class Network {
    constructor(nbIn, nb1, nb2) {
        this.layers = [];
        this.layers.push(this.createLayer(nbIn, nb1))
        this.layers.push(this.createLayer(nb1, 1))
        // this.layers.push(this.createLayer(nb2, 1))
    }

    createLayer(nbParent, nbNeurons) {
        let layer = [];
        for (let i = 0; i < nbNeurons; i++) {
            layer.push(new Neuron(nbParent));
        }
        return layer;
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

    backPropagation(target) {
        const lastLayer = this.layers[this.layers.length - 1];
        for (const r of lastLayer) {
            const prevLayer = this.layers[this.layers.length - 2];
            for (let i = 0; i < prevLayer.length; i++) {
                const prevNeuron = prevLayer[i];
                let derivedPrevWeightCost = this.derivedCost(r, prevNeuron, target);
                r.weights[i] += derivedPrevWeightCost * velocity;
            }
        }
    }

    derivedCost(result, prevNeuron, target) {
        const dC = 2 * (result.activation - target);
        const dS = derivedSigmoid(prevNeuron.sum);
        const prevActivation = prevNeuron.activation;
        return dC * dS * prevActivation;
    }



}


const network = new Network(datas[0].values.length, 16, 16);

let result;

for (let i= 0; i < 10; i++) {

    
    console.log(datas[i%datas.length].values);
    result = network.activate(datas[i%datas.length].values);
    console.log(result, 'should be => '+datas[i%datas.length].target);
    network.backPropagation(datas[i%datas.length].target)
}



// console.log(datas[0].values);
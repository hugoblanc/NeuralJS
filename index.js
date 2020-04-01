const chalk = require('chalk');


const rl = require('readline');

function clearLine(dist) {
    rl.cursorTo(process.stdout, dist);
    rl.clearLine(process.stdout, dist);
}



const datas = [
    { values: [1,0,1,1,0], target: [1, 0] },
    { values: [0,1,1,1,1], target: [1, 0] },
    { values: [1,0,1,0,1], target: [1, 0] },
    { values: [1,0,0,0,1], target: [0, 1] },
    { values: [1,0,1,0,0], target: [0, 1] },
    { values: [0,1,0,1,0], target: [0, 1] },
    // { values: [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1], target: [0, 1] },
    // { values: [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0], target: [1, 0] },
    // { values: [0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], target: [1, 0] },
    // { values: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1], target: [0, 1] },
];
const velocity = 0.82;


// Last ten precision values




// COUNTER - ANALYTICS

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

function reLU(x) {
    return (x > 0) ? x : 0;
}

function derivedReLU(x) {
    return (x > 0) ? 1 : 0;
}

function activationFunction(x) {
    return sigmoid(x);
}

function derivedActivationFunction(x) {
    return derivedSigmoid(x);
}


class Neuron {
    constructor(nbParents, parentLayer = []) {
        // Init synapses
        this.synapses = [];
        this.sum = 0;
        this.activation = 0;
        this.childSynapses = [];
        this.bias = 0;
        for (let i = 0; i < nbParents; i++) {
            const parentNeuron = parentLayer[i] ? parentLayer[i] : null;
            this.synapses.push({ weight: parentNeuron == null ? 1 : (Math.random() - 1) * 2, parentNeuron });
            if (parentNeuron) {
                parentNeuron.childSynapses.push(this);
            }
        }
    }


    toString() {
        console.log(`synapses:${this.synapses.map((s, i) => 'w' + i + ' ' + s.weight).join('--')} sum:${this.sum} activation:${this.activation} bias:${this.bias}`);

    }

    activate(inputs) {
        let total = 0;
        for (let i = 0; i < inputs.length; i++) {
            const weight = this.synapses[i] ? this.synapses[i].weight : 1;

            const wN = inputs[i] * weight;
            total += wN;
        }

        this.sum = total + this.bias;
        this.activation = activationFunction(total);

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

            this.layers.push(this.createLayer(prevLayer, nextLayer))
        }
    }

    toString() {
        for (let i = 0; i < this.layers.length; i++) {
            const layer = this.layers[i];
            console.log("-----------------------------------");
            console.log("Layer n° " + i);
            layer.map((n, i) => 'Neuron n°' + i + ' ' + n.toString()).join('\n');
        }
    }

    createLayer(nbParent, nbNeurons) {
        let layer = [];
        for (let i = 0; i < nbNeurons; i++) {
            const prevLayer = this.layers.length > 0 ? this.layers[this.layers.length - 1] : undefined;
            layer.push(new Neuron(nbParent, prevLayer));
        }
        return layer;
    }

    activate(inputs, layerIndex = 0) {

        if (layerIndex === this.layers.length) {
            return inputs;
        }



        let accFuncResults = [];
        for (const neuron of this.layers[layerIndex]) {
            accFuncResults.push(neuron.activate(inputs));
        }
        if (layerIndex === 0) {
            accFuncResults = inputs;
        }
        return this.activate(accFuncResults, ++layerIndex);
    }


    backPropagationBis(target) {
        const lastLayer = this.layers[this.layers.length - 1];

        let count = 0;
        for (const finalNeuron of lastLayer) {
            const dC = 2 * (finalNeuron.activation - target[count]);
            for (const synapse of finalNeuron.synapses) {
                localRecursive = 0;
                const dCostPZ = derivedActivationFunction(synapse.parentNeuron.sum) * dC;
                this.recPropagation(dC, finalNeuron, dCostPZ, synapse);
                localRecursive = 0;
            }
            count++;
        }
    }


    recPropagation(dC, finalNeuron, dCostPZ, ...synapses) {
        globRecursiveCounter++;
        const youngestSynapse = synapses[synapses.length - 1];


        if (youngestSynapse.parentNeuron == null) return;
        const parentNeuron = youngestSynapse.parentNeuron;

        youngestSynapse.bias -= (dCostPZ) * (velocity);
        youngestSynapse.weight -= (dCostPZ * parentNeuron.activation) * (velocity);

        if (parentNeuron.synapses == null) return;
        for (let i = 0; i < parentNeuron.synapses.length; i++) {
            const pSynapse = parentNeuron.synapses[i];
            if (pSynapse.parentNeuron !== null) {
                dCostPZ = derivedActivationFunction(pSynapse.parentNeuron.sum) * youngestSynapse.weight * dCostPZ;
                this.recPropagation(dC, finalNeuron, dCostPZ, ...synapses, pSynapse)
            }
        }
    }


}


const network = new Network(datas[0].values.length, 150, datas[0].target.length);


const precision = [];
trainNetwork();



function showPrecision(pValue, target, result) {
    const percent = pValue * 100;
    if (precision.length > 10) {
        precision.unshift();
    }
    precision.push(percent);

    let total = 0;
    for (let p of precision) {
        total += p;
    }
    total = total / precision.length;

    let line = `Precision: ${Number.parseFloat(total).toFixed(3)}%  Target => ${Number.parseFloat(target[0]).toFixed(3)} Result => ${Number.parseFloat(result[0]).toFixed(3)} [`
    let i = 0;
    while (i < 100) {
        if (i % 2 === 0) {
            if (i < total) {
                line += '>';
            } else {
                line += ' ';
            }
        }
        i++;
    }
    line += ']  Activation: ' + globActivateCounter;

    line = chalk.bgRgb(Math.ceil(240 - 150 * (total / 100)), Math.ceil(130 * (total / 100)), 0)(line)

    return line;
}






function trainNetwork() {
    for (let i = 0; i < 100000000; i++) {
        const currentData = datas[i % datas.length];
        const target = currentData.target;
        const values = currentData.values;

        const result = network.activate(values);


        let totalPrecision = 0;
        for (let i = 0; i < target.length; i++) {
            const cV = target[i];
            const cR = result[i];

            totalPrecision += 1 - Math.abs(cV - cR);
        }
        totalPrecision = totalPrecision / target.length;

        if (i % 3 === 0) {
            const strPrecision = showPrecision(totalPrecision, target, result);
            clearLine(0)
            process.stdout.write(strPrecision);
        }

        network.backPropagationBis(target)
    }

}



console.log("Neuron activation" + globActivateCounter);
console.log("BackProp recursive call" + globRecursiveCounter);
console.log("Cost derived calculation" + globCostCalcul);










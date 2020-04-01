const chalk = require('chalk');

const dataSet1 = require('./dataset/dataset1')
const dataSet2 = require('./dataset/dataset2')
const dataSet3 = require('./dataset/dataset3')

const rl = require('readline');

function clearLine(dist) {
    rl.cursorTo(process.stdout, dist);
    rl.clearLine(process.stdout, dist);
}


const MODE_DEBUG = 'DEBUG';
const MODE_PRESENTATION = 'PRESENTATION';


let mode = MODE_DEBUG;
mode = MODE_PRESENTATION;






// const datas = dataSet1;
// const datas = dataSet2;
const datas = dataSet3;

let velocity = 13.820002;
let lastTotalPrecision = 0;


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
        this.bias = 1;
        for (let i = 0; i < nbParents; i++) {
            const parentNeuron = parentLayer[i] ? parentLayer[i] : null;
            this.synapses.push({ weight: parentNeuron == null ? 1 : (Math.random() - 1) * 2, parentNeuron });
            if (parentNeuron) {
                parentNeuron.childSynapses.push(this);
            }
        }
    }


    toString() {
        return `synapses:${this.synapses.map((s, i) => 'w' + i + ' ' + Number.parseFloat(s.weight).toFixed(6)).join('--')} sum:${Number.parseFloat(this.sum).toFixed(6)} activation:${Number.parseFloat(this.activation).toFixed(6)} bias:${Number.parseFloat(this.bias).toFixed(6)}`;

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
        let str = '';
        for (let i = 0; i < this.layers.length; i++) {
            const layer = this.layers[i];
            str += '\n \n';
            str += "-------------------------------------------------------------------------------------------------------------------------------------------- \n";
            str += "Layer n° " + i + '\n';
            str += layer.map((n, i) => 'Neuron n°' + i + ' ' + n.toString()).join('\n');
        }
        return str;
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
                this.recPropagation(dC, dCostPZ, synapse);
                localRecursive = 0;
            }
            count++;
        }
    }


    recPropagation(dC, dCostPZ, ...synapses) {
        globRecursiveCounter++;
        const youngestSynapse = synapses[synapses.length - 1];


        if (youngestSynapse.parentNeuron == null) return;
        const parentNeuron = youngestSynapse.parentNeuron;
        parentNeuron.bias -= (dCostPZ) * (velocity);
        youngestSynapse.weight -= (dCostPZ * parentNeuron.activation) * (velocity);

        if (parentNeuron.synapses == null) return;
        for (let i = 0; i < parentNeuron.synapses.length; i++) {
            const pSynapse = parentNeuron.synapses[i];
            if (pSynapse.parentNeuron !== null) {
                this.recPropagation(dC, derivedActivationFunction(pSynapse.parentNeuron.sum) * youngestSynapse.weight * dCostPZ, ...synapses, pSynapse)
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
        precision.shift();
    }
    
    precision.push(percent);

    let total = 0;
    for (let p of precision) {
        total += p;
    }
    total = total / precision.length;

    let line = `Precision: ${Number.parseFloat(total).toFixed(6)}%   T => ${target.map(t => Number.parseFloat(t).toFixed(0)).join(' ') } R => ${ result.map(r => Number.parseFloat(r).toFixed(3)).join(' ')} [`
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
    line += ']  Activation: ' + globActivateCounter + ' Recursive call ' + globRecursiveCounter + ' Velocity ' + velocity;

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

            totalPrecision += (1 - Math.abs(cV - cR));
        }
        console.log(totalPrecision);
        
        totalPrecision = totalPrecision / target.length;

        // if(lastTotalPrecision < totalPrecision){
        //     velocity *= 1.2;
        // } else {
        //     velocity /= 1.1;
        //     // velocity *= 2;
        // }

        // lastTotalPrecision = totalPrecision;

        if (mode === MODE_PRESENTATION) {
            // if (i % 3 === 0) {
                const strPrecision = showPrecision(totalPrecision, target, result);
                clearLine(0)
                process.stdout.write(strPrecision);
            // }
        } else {
            console.log(network.toString());
        }


        network.backPropagationBis(target)
        // console.log(network.toString());
    }

}



console.log("Neuron activation" + globActivateCounter);
console.log("BackProp recursive call" + globRecursiveCounter);
console.log("Cost derived calculation" + globCostCalcul);










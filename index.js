const chalk = require('chalk');
const rl = require('readline');

const DATASETS = {
    '1':       { file: './dataset/dataset1',           desc: 'Simple binary (2 in, 1 out)' },
    '2':       { file: './dataset/dataset2',           desc: 'Multi-feature (5 in, 2 out)' },
    '3':       { file: './dataset/dataset3',           desc: 'Dense pattern (16 in, 2 out)' },
    'xor':     { file: './dataset/dataset4_xor',       desc: 'XOR gate' },
    'parity':  { file: './dataset/dataset5_parity',    desc: 'Parity check (4 bits)' },
    'digits':  { file: './dataset/dataset6_digits',    desc: 'Digit recognition 0-4 (3x5 grid)' },
    'add':     { file: './dataset/dataset7_binary_add', desc: 'Binary addition (2+2 bits)' },
    'auto':    { file: './dataset/dataset8_autoencoder', desc: 'Autoencoder 8->3->8 (use --layers 8,5,3,5,8)' },
};

const args = process.argv.slice(2);
const flags = {};
for (let i = 0; i < args.length; i++) {
    if (args[i].startsWith('--')) {
        const key = args[i].slice(2);
        flags[key] = args[i + 1] || true;
        i++;
    } else if (!flags.dataset) {
        flags.dataset = args[i];
    }
}

if (flags.help || flags.h) {
    console.log(`Usage: node index.js [dataset] [options]

Datasets: ${Object.entries(DATASETS).map(([k, v]) => `\n  ${k.padEnd(10)} ${v.desc}`).join('')}

Options:
  --lr <n>          Learning rate (default: 0.5)
  --iterations <n>  Max iterations (default: 50000)
  --hidden <n>      Hidden layer size (default: 8)
  --layers <n,n,..> Custom layer sizes (e.g. 8,5,3,5,8 for autoencoder)
  --debug           Show full network state instead of progress bar
  --inspect         After training, show bottleneck activations per sample
  --help            Show this help`);
    process.exit(0);
}

const datasetKey = flags.dataset || 'xor';
if (!DATASETS[datasetKey]) {
    console.error(`Unknown dataset "${datasetKey}". Available: ${Object.keys(DATASETS).join(', ')}`);
    process.exit(1);
}
const datas = require(DATASETS[datasetKey].file);
console.log(`Dataset: ${datasetKey} - ${DATASETS[datasetKey].desc}`);

let velocity = parseFloat(flags.lr) || 0.5;
const MAX_ITERATIONS = parseInt(flags.iterations) || 50000;
const HIDDEN_SIZE = parseInt(flags.hidden) || 8;
const mode = flags.debug ? 'DEBUG' : 'PRESENTATION';

function clearLine(dist) {
    rl.cursorTo(process.stdout, dist);
    rl.clearLine(process.stdout, dist);
}

let globActivateCounter = 0;
let globRecursiveCounter = 0;

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
            this.synapses.push({ weight: parentNeuron == null ? 1 : (Math.random() * 2) - 1, parentNeuron });
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
        this.activation = activationFunction(this.sum);

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

    activate(inputs) {
        const inputLayer = this.layers[0];
        for (let i = 0; i < inputLayer.length; i++) {
            inputLayer[i].activation = inputs[i];
            inputLayer[i].sum = inputs[i];
        }

        for (let l = 1; l < this.layers.length; l++) {
            for (const neuron of this.layers[l]) {
                let total = 0;
                for (const synapse of neuron.synapses) {
                    total += synapse.parentNeuron.activation * synapse.weight;
                }
                neuron.sum = total + neuron.bias;
                neuron.activation = activationFunction(neuron.sum);
            }
        }

        return this.layers[this.layers.length - 1].map(n => n.activation);
    }


    backPropagate(target) {
        const outputLayer = this.layers[this.layers.length - 1];

        for (let i = 0; i < outputLayer.length; i++) {
            const neuron = outputLayer[i];
            const dCda = 2 * (neuron.activation - target[i]);
            neuron.delta = dCda * derivedActivationFunction(neuron.sum);
        }

        for (let l = this.layers.length - 2; l >= 1; l--) {
            for (let j = 0; j < this.layers[l].length; j++) {
                const neuron = this.layers[l][j];
                let errorSum = 0;
                for (const nextNeuron of this.layers[l + 1]) {
                    errorSum += nextNeuron.synapses[j].weight * nextNeuron.delta;
                }
                neuron.delta = errorSum * derivedActivationFunction(neuron.sum);
            }
        }

        for (let l = 1; l < this.layers.length; l++) {
            for (const neuron of this.layers[l]) {
                neuron.bias -= neuron.delta * velocity;
                for (const synapse of neuron.synapses) {
                    synapse.weight -= neuron.delta * synapse.parentNeuron.activation * velocity;
                }
            }
        }
    }


}


let layerSizes;
if (flags.layers) {
    layerSizes = flags.layers.split(',').map(Number);
    if (layerSizes[0] !== datas[0].values.length) {
        console.error(`First layer (${layerSizes[0]}) must match input size (${datas[0].values.length})`);
        process.exit(1);
    }
    if (layerSizes[layerSizes.length - 1] !== datas[0].target.length) {
        console.error(`Last layer (${layerSizes[layerSizes.length - 1]}) must match target size (${datas[0].target.length})`);
        process.exit(1);
    }
} else {
    layerSizes = [datas[0].values.length, HIDDEN_SIZE, datas[0].target.length];
}
console.log(`Network: ${layerSizes.join(' -> ')}`);
const network = new Network(...layerSizes);


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
    for (let i = 0; i < MAX_ITERATIONS; i++) {
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
        totalPrecision = totalPrecision / target.length;

        // if(lastTotalPrecision < totalPrecision){
        //     velocity *= 1.2;
        // } else {
        //     velocity /= 1.1;
        //     // velocity *= 2;
        // }

        // lastTotalPrecision = totalPrecision;

        if (mode === 'PRESENTATION') {
            // if (i % 3 === 0) {
                const strPrecision = showPrecision(totalPrecision, target, result);
                clearLine(0)
                process.stdout.write(strPrecision);
            // }
        } else {
            console.log(network.toString());
        }


        network.backPropagate(target)
        // console.log(network.toString());
    }

}



console.log("\nNeuron activations: " + globActivateCounter);

if (flags.inspect) {
    const bottleneckIndex = Math.floor(network.layers.length / 2);
    console.log(`\n--- Bottleneck inspection (layer ${bottleneckIndex}, ${network.layers[bottleneckIndex].length} neurons) ---\n`);
    for (const sample of datas) {
        network.activate(sample.values);
        const bottleneck = network.layers[bottleneckIndex].map(n => n.activation);
        const rounded = bottleneck.map(v => Math.round(v));
        const output = network.layers[network.layers.length - 1].map(n => n.activation);
        console.log(
            `Input: [${sample.values.join('')}]` +
            `  Bottleneck: [${bottleneck.map(v => v.toFixed(3)).join(', ')}]` +
            `  ~ [${rounded.join('')}]` +
            `  Output: [${output.map(v => v.toFixed(3)).join(', ')}]`
        );
    }
}










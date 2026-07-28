# Neural Network from Scratch

Build a neural network from zero, step by step, in TypeScript.
No libraries, no magic - just the math, made tangible.

## Why this exists

I learned how neural networks work by watching [3Blue1Brown's deep learning series](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi).
Those videos are the best theoretical explanation I have found - go watch them first if you haven't.

But after watching, I wanted to **build one myself** to check that I had actually understood.
This repo is the result: a complete neural network implemented from scratch, with a step-by-step tutorial that follows the same progression.

The approach here is practical.
Each step is a standalone file you can run, modify, and break.
You see the effect of every concept immediately.

## Quick start

```bash
git clone https://github.com/hugoblanc/NeuralJS.git
cd NeuralJS
npm install
```

Run any step directly:

```bash
npx tsx steps/01-neuron.ts
npx tsx steps/02-forward.ts
npx tsx steps/03-cost.ts
npx tsx steps/04-backprop.ts
npx tsx steps/05-training.ts
npx tsx steps/06-digits.ts        # requires: npm run download-mnist
npx tsx steps/07-autoencoder.ts
```

## The object model

A neural network maps onto objects almost too neatly, and this tutorial leans on that.
The class hierarchy is not decoration - it is the explanation.

```
Network   ordered list of layers      forward(), backpropagate()
  Layer   group of neurons            activate(), computeHiddenDeltas()
 Neuron   holds one number in [0,1]   activate(), applyGradient()
Synapse   weighted connection         source, target, weight
```

Every step redefines these classes with a little more in them than the step before.
That growth is the curriculum: what step 04 adds on top of step 03 *is* backpropagation.
Each file is **self-contained** - no imports from the rest of the project, so you can open any one of them and read it top to bottom.

The single most useful thing to notice: `Network.forward()` is three lines and `Network.backpropagate()` is three steps.
Everything else happens inside individual neurons that never see more than their immediate neighbours.

## The steps

### [01 - The Neuron](steps/01-neuron.ts)
A neuron is one number between 0 and 1.
It reads its neighbours through weighted synapses, sums, adds a bias, and squishes the result through sigmoid.
Builds `Synapse` and `Neuron`, and wires four of them by hand.
> 3Blue1Brown: [Chapter 1, 0:00-5:00](https://www.youtube.com/watch?v=aircAruvnKk)

### [02 - The Forward Pass](steps/02-forward.ts)
Wiring neurons by hand does not scale, so: `Layer` and `Network`.
Data flows from input to output. With random weights the output means nothing, but the plumbing works.
> 3Blue1Brown: [Chapter 1, 5:00-15:00](https://www.youtube.com/watch?v=aircAruvnKk&t=300)

### [03 - The Cost Function](steps/03-cost.ts)
How do you put a number on "this network is bad"?
Introduces `CostFunction` with its two faces: `evaluate()` for us, `gradient()` for step 04.
Then sweeps a single weight to show the cost landscape, and why one weight at a time is hopeless.
> 3Blue1Brown: [Chapter 2, 0:00-6:00](https://www.youtube.com/watch?v=IHZwWFHWa-w)

### [04 - Backpropagation](steps/04-backprop.ts)
The core algorithm, and the one new thing the object graph needs: synapses that know both ends.
Forward uses `neuron.inputs`, backward uses `neuron.outputs`.
Blame flows right to left, with each neuron talking only to its neighbours.
Watch a single backward pass in slow motion, then watch it learn XOR.
> 3Blue1Brown: [Chapter 3](https://www.youtube.com/watch?v=Ilg3gGewQ5U)

### [05 - The Training Loop](steps/05-training.ts)
Epochs, learning rates and progress bars are not the network's job, so they get their own objects: `Dataset`, `Trainer`, `ProgressBar`, `ProgressMetric`.
```bash
npx tsx steps/05-training.ts xor 0.5         # default
npx tsx steps/05-training.ts xor 0.005       # too slow to get there
npx tsx steps/05-training.ts xor 30          # broken: stuck at cost 0.5
npx tsx steps/05-training.ts add 0.5 60000   # it has to invent a carry
```

### [06 - Recognizing Digits](steps/06-digits.ts)
MNIST: 784 pixels in, a digit 0-9 out - the same problem 3Blue1Brown uses throughout the series.
This is the payoff of everything above: `Neuron`, `Layer` and `Network` are copied from step 05 **unchanged**.
Only the objects around the data are new (`Digit`, `MnistLoader`, `Classifier`, `Evaluation`).
```bash
npm run download-mnist
npx tsx steps/06-digits.ts 500 20000      # fast, roughly 80%
npx tsx steps/06-digits.ts 2000 100000    # slower, roughly 90%
```

### [07 - The Autoencoder](steps/07-autoencoder.ts)
Force 8 patterns through a bottleneck of 3 neurons and train the network to reproduce its own input.
`Autoencoder extends Network` - same maths, new vocabulary: `encode()`, `decode()`, `bottleneckLayer`.

The obvious guess is that it invents binary: 2^3 = 8, exactly enough.
It doesn't, and finding out why is the point of the step.
A bottleneck neuron holds a real number, so the code is a point in a cube rather than a 3-bit word - which is exactly what an embedding is.
The step ends by asking you to shrink the bottleneck to 2 neurons: if the code were really binary that would leave 4 slots for 8 patterns and it would have to fail, yet it usually still recovers all 8.

## The full library

Beyond the tutorial, `src/` contains the same classes as a modular implementation:

```bash
npm run build
node dist/index.js --help
node dist/index.js xor
node dist/index.js mnist --samples 1000 --show 10
node dist/index.js auto --inspect
```

## Don't take my word for it

Backpropagation is easy to get subtly wrong in a way that still trains and still looks convincing.
The honest test is to compare its gradients against numerically estimated ones:

```bash
npm run gradcheck
```

It nudges every weight and bias by 1e-5, measures how the cost actually moves, and compares that to what backpropagation claimed.
The two agree to about 1e-11.

## Requirements

- Node.js 18+
- TypeScript (included in dependencies)
- ~55MB disk for MNIST data (optional, only for step 06)

## References

- [3Blue1Brown - Neural Networks](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) - the theoretical foundation
- [Andrej Karpathy - Let's build](https://www.youtube.com/@AndrejKarpathy) - the "build from scratch" inspiration
- [Michael Nielsen - Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/) - the textbook

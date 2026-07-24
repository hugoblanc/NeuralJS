# Neural Network from Scratch

Build a neural network from zero, step by step, in TypeScript.
No libraries, no magic - just the math, made tangible.

## Why this exists

I learned how neural networks work by watching [3Blue1Brown's deep learning series](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi).
These videos are the best theoretical explanation I've found - go watch them first if you haven't.

But after watching, I wanted to **build one myself** to make sure I actually understood.
This repo is the result: a complete neural network implemented from scratch, with a step-by-step tutorial that mirrors the progression of those videos.

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

## The steps

Each file is **self-contained** - it includes all the code it needs, no imports from the rest of the project.
You can open any file and read it top to bottom.
The code grows progressively: what step 04 adds on top of step 03 is exactly what backpropagation is.

### [01 - The Neuron](steps/01-neuron.ts)
A neuron is just a number between 0 and 1.
It takes weighted inputs, sums them, and squishes the result through sigmoid.
> 3Blue1Brown: [Chapter 1, 0:00-5:00](https://www.youtube.com/watch?v=aircAruvnKk)

### [02 - The Forward Pass](steps/02-forward.ts)
Stack neurons into layers, connect every neuron to the previous layer.
Data flows from input to output - this is the forward pass.
With random weights, the network outputs nonsense.
> 3Blue1Brown: [Chapter 1, 5:00-15:00](https://www.youtube.com/watch?v=aircAruvnKk&t=300)

### [03 - The Cost Function](steps/03-cost.ts)
How do you measure "this network is bad"?
Sum of squared differences between output and target. That's the cost.
Our goal: make this number as small as possible.
> 3Blue1Brown: [Chapter 2, 0:00-6:00](https://www.youtube.com/watch?v=IHZwWFHWa-w)

### [04 - Backpropagation](steps/04-backprop.ts)
The core algorithm. For each weight, figure out how much it contributed to the error, then nudge it the other way.
The error flows backward through the network, layer by layer.
Watch the network learn XOR from scratch.
> 3Blue1Brown: [Chapter 3](https://www.youtube.com/watch?v=Ilg3gGewQ5U)

### [05 - The Training Loop](steps/05-training.ts)
Put it all together with a live progress bar.
Experiment with the learning rate - too small and it's slow, too big and it explodes.
```bash
npx tsx steps/05-training.ts xor 0.5     # default
npx tsx steps/05-training.ts xor 5.0     # too fast?
npx tsx steps/05-training.ts parity 0.5  # harder problem
```

### [06 - Recognizing Digits](steps/06-digits.ts)
The real test: MNIST handwritten digits.
784 pixels go in, a digit 0-9 comes out.
The exact same problem 3Blue1Brown uses throughout his series.
```bash
npm run download-mnist
npx tsx steps/06-digits.ts 1000 50000    # 1000 samples, 50k iterations
```

### [07 - The Autoencoder](steps/07-autoencoder.ts)
Force 8 patterns through a bottleneck of 3 neurons.
The network invents binary encoding on its own - nobody told it about binary.
This is the foundation of representation learning.

## The full library

Beyond the tutorial, `src/` contains a clean, modular implementation:

```bash
npm run build
node dist/index.js --help
node dist/index.js xor
node dist/index.js mnist --samples 1000 --show 10
node dist/index.js auto --inspect
```

## Requirements

- Node.js 18+
- TypeScript (included in dependencies)
- ~55MB disk for MNIST data (optional, only for step 06)

## References

- [3Blue1Brown - Neural Networks](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) - the theoretical foundation
- [Andrej Karpathy - Let's build](https://www.youtube.com/@AndrejKarpathy) - the "build from scratch" inspiration
- [Michael Nielsen - Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/) - the textbook

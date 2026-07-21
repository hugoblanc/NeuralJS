export interface Sample {
  inputs: number[];
  target: number[];
}

export interface Dataset {
  description: string;
  samples: Sample[];
  recommendedLayers?: number[];
}

export const DATASETS: Record<string, Dataset> = {

  xor: {
    description: 'XOR gate - non-linearly separable, needs a hidden layer',
    samples: [
      { inputs: [0, 0], target: [0] },
      { inputs: [0, 1], target: [1] },
      { inputs: [1, 0], target: [1] },
      { inputs: [1, 1], target: [0] },
    ],
  },

  parity: {
    description: 'Parity check - is the count of 1s odd or even? (4 bits)',
    samples: [
      { inputs: [0, 0, 0, 0], target: [0] },
      { inputs: [1, 0, 0, 0], target: [1] },
      { inputs: [0, 1, 0, 0], target: [1] },
      { inputs: [0, 0, 1, 0], target: [1] },
      { inputs: [0, 0, 0, 1], target: [1] },
      { inputs: [1, 1, 0, 0], target: [0] },
      { inputs: [1, 0, 1, 0], target: [0] },
      { inputs: [1, 0, 0, 1], target: [0] },
      { inputs: [0, 1, 1, 0], target: [0] },
      { inputs: [0, 1, 0, 1], target: [0] },
      { inputs: [0, 0, 1, 1], target: [0] },
      { inputs: [1, 1, 1, 0], target: [1] },
      { inputs: [1, 1, 0, 1], target: [1] },
      { inputs: [1, 0, 1, 1], target: [1] },
      { inputs: [0, 1, 1, 1], target: [1] },
      { inputs: [1, 1, 1, 1], target: [0] },
    ],
  },

  digits: {
    description: 'Digit recognition 0-4 on a 3x5 pixel grid (one-hot output)',
    recommendedLayers: [15, 16, 5],
    samples: [
      // 0:  ###  #.#  #.#  #.#  ###
      { inputs: [1,1,1, 1,0,1, 1,0,1, 1,0,1, 1,1,1], target: [1,0,0,0,0] },
      // 1:  .#.  .#.  .#.  .#.  .#.
      { inputs: [0,1,0, 0,1,0, 0,1,0, 0,1,0, 0,1,0], target: [0,1,0,0,0] },
      // 2:  ###  ..#  ###  #..  ###
      { inputs: [1,1,1, 0,0,1, 1,1,1, 1,0,0, 1,1,1], target: [0,0,1,0,0] },
      // 3:  ###  ..#  ###  ..#  ###
      { inputs: [1,1,1, 0,0,1, 1,1,1, 0,0,1, 1,1,1], target: [0,0,0,1,0] },
      // 4:  #.#  #.#  ###  ..#  ..#
      { inputs: [1,0,1, 1,0,1, 1,1,1, 0,0,1, 0,0,1], target: [0,0,0,0,1] },
    ],
  },

  add: {
    description: 'Binary addition - A(2 bits) + B(2 bits) = S(3 bits)',
    samples: [
      { inputs: [0,0, 0,0], target: [0,0,0] }, // 0+0=0
      { inputs: [0,0, 0,1], target: [0,0,1] }, // 0+1=1
      { inputs: [0,0, 1,0], target: [0,1,0] }, // 0+2=2
      { inputs: [0,0, 1,1], target: [0,1,1] }, // 0+3=3
      { inputs: [0,1, 0,0], target: [0,0,1] }, // 1+0=1
      { inputs: [0,1, 0,1], target: [0,1,0] }, // 1+1=2
      { inputs: [0,1, 1,0], target: [0,1,1] }, // 1+2=3
      { inputs: [0,1, 1,1], target: [1,0,0] }, // 1+3=4
      { inputs: [1,0, 0,0], target: [0,1,0] }, // 2+0=2
      { inputs: [1,0, 0,1], target: [0,1,1] }, // 2+1=3
      { inputs: [1,0, 1,0], target: [1,0,0] }, // 2+2=4
      { inputs: [1,0, 1,1], target: [1,0,1] }, // 2+3=5
      { inputs: [1,1, 0,0], target: [0,1,1] }, // 3+0=3
      { inputs: [1,1, 0,1], target: [1,0,0] }, // 3+1=4
      { inputs: [1,1, 1,0], target: [1,0,1] }, // 3+2=5
      { inputs: [1,1, 1,1], target: [1,1,0] }, // 3+3=6
    ],
  },

  auto: {
    description: 'Autoencoder - compress 8 one-hot inputs through 3 neurons, then reconstruct',
    recommendedLayers: [8, 5, 3, 5, 8],
    samples: [
      { inputs: [1,0,0,0,0,0,0,0], target: [1,0,0,0,0,0,0,0] },
      { inputs: [0,1,0,0,0,0,0,0], target: [0,1,0,0,0,0,0,0] },
      { inputs: [0,0,1,0,0,0,0,0], target: [0,0,1,0,0,0,0,0] },
      { inputs: [0,0,0,1,0,0,0,0], target: [0,0,0,1,0,0,0,0] },
      { inputs: [0,0,0,0,1,0,0,0], target: [0,0,0,0,1,0,0,0] },
      { inputs: [0,0,0,0,0,1,0,0], target: [0,0,0,0,0,1,0,0] },
      { inputs: [0,0,0,0,0,0,1,0], target: [0,0,0,0,0,0,1,0] },
      { inputs: [0,0,0,0,0,0,0,1], target: [0,0,0,0,0,0,0,1] },
    ],
  },
};

/**
 * A cost function has two faces that must always agree: one for us to
 * read, one for the gradients. Keeping them in one object means the
 * network can be handed a different cost without changing.
 */
export interface CostFunction {
  readonly name: string;

  /** How wrong is this output? Lower is better, 0 is perfect. */
  evaluate(output: number[], target: number[]): number;

  /**
   * For each output neuron: how does the cost move when that neuron's
   * activation goes up? Positive means "too high, come down".
   */
  gradient(output: number[], target: number[]): number[];
}

/** cost = average of (output - target)^2 over the output neurons. */
export class MeanSquaredError implements CostFunction {
  readonly name = 'mean squared error';

  evaluate(output: number[], target: number[]): number {
    let total = 0;
    for (let i = 0; i < target.length; i++) {
      const error = output[i] - target[i];
      total += error * error;
    }
    return total / target.length;
  }

  gradient(output: number[], target: number[]): number[] {
    // d/da of (a - t)^2 is 2 * (a - t). The 1/n from evaluate() is dropped
    // on purpose: it is a constant factor the learning rate absorbs.
    return target.map((t, i) => 2 * (output[i] - t));
  }
}

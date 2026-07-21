export interface ActivationFn {
  apply(x: number): number;
  derivative(x: number): number;
}

export const sigmoid: ActivationFn = {
  apply(x) {
    return 1 / (1 + Math.exp(-x));
  },
  derivative(x) {
    const s = sigmoid.apply(x);
    return s * (1 - s);
  },
};

export const relu: ActivationFn = {
  apply(x) {
    return x > 0 ? x : 0;
  },
  derivative(x) {
    return x > 0 ? 1 : 0;
  },
};

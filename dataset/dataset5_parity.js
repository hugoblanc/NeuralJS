// Parity check : le nombre de 1 est-il pair (0) ou impair (1) ?
const datas = [
    { values: [0, 0, 0, 0], target: [0] },
    { values: [1, 0, 0, 0], target: [1] },
    { values: [0, 1, 0, 0], target: [1] },
    { values: [0, 0, 1, 0], target: [1] },
    { values: [0, 0, 0, 1], target: [1] },
    { values: [1, 1, 0, 0], target: [0] },
    { values: [1, 0, 1, 0], target: [0] },
    { values: [1, 0, 0, 1], target: [0] },
    { values: [0, 1, 1, 0], target: [0] },
    { values: [0, 1, 0, 1], target: [0] },
    { values: [0, 0, 1, 1], target: [0] },
    { values: [1, 1, 1, 0], target: [1] },
    { values: [1, 1, 0, 1], target: [1] },
    { values: [1, 0, 1, 1], target: [1] },
    { values: [0, 1, 1, 1], target: [1] },
    { values: [1, 1, 1, 1], target: [0] },
]

module.exports = datas;

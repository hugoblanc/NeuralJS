// Addition binaire 2 bits : A(2 bits) + B(2 bits) = S(3 bits)
// Input: [a1, a0, b1, b0]   Output: [s2, s1, s0]
const datas = [
    { values: [0,0, 0,0], target: [0,0,0] }, // 0+0=0
    { values: [0,0, 0,1], target: [0,0,1] }, // 0+1=1
    { values: [0,0, 1,0], target: [0,1,0] }, // 0+2=2
    { values: [0,0, 1,1], target: [0,1,1] }, // 0+3=3
    { values: [0,1, 0,0], target: [0,0,1] }, // 1+0=1
    { values: [0,1, 0,1], target: [0,1,0] }, // 1+1=2
    { values: [0,1, 1,0], target: [0,1,1] }, // 1+2=3
    { values: [0,1, 1,1], target: [1,0,0] }, // 1+3=4
    { values: [1,0, 0,0], target: [0,1,0] }, // 2+0=2
    { values: [1,0, 0,1], target: [0,1,1] }, // 2+1=3
    { values: [1,0, 1,0], target: [1,0,0] }, // 2+2=4
    { values: [1,0, 1,1], target: [1,0,1] }, // 2+3=5
    { values: [1,1, 0,0], target: [0,1,1] }, // 3+0=3
    { values: [1,1, 0,1], target: [1,0,0] }, // 3+1=4
    { values: [1,1, 1,0], target: [1,0,1] }, // 3+2=5
    { values: [1,1, 1,1], target: [1,1,0] }, // 3+3=6
]

module.exports = datas;

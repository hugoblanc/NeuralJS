// Autoencoder 8->3->8 : compresser 8 bits one-hot en 3 bits puis reconstruire
// Le réseau doit découvrir le codage binaire tout seul
// Architecture recommandée : --layers 8,5,3,5,8
const datas = [
    { values: [1,0,0,0,0,0,0,0], target: [1,0,0,0,0,0,0,0] },
    { values: [0,1,0,0,0,0,0,0], target: [0,1,0,0,0,0,0,0] },
    { values: [0,0,1,0,0,0,0,0], target: [0,0,1,0,0,0,0,0] },
    { values: [0,0,0,1,0,0,0,0], target: [0,0,0,1,0,0,0,0] },
    { values: [0,0,0,0,1,0,0,0], target: [0,0,0,0,1,0,0,0] },
    { values: [0,0,0,0,0,1,0,0], target: [0,0,0,0,0,1,0,0] },
    { values: [0,0,0,0,0,0,1,0], target: [0,0,0,0,0,0,1,0] },
    { values: [0,0,0,0,0,0,0,1], target: [0,0,0,0,0,0,0,1] },
]

module.exports = datas;

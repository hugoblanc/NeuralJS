// Mini-digits 3x5 : reconnaitre les chiffres 0-4 sur une grille 3x5
// Chaque chiffre est encodé en 15 pixels (3 colonnes x 5 lignes)
// Output : one-hot encoding [0, 1, 2, 3, 4]
const datas = [
    // 0 :  ###    ###    ###    ###    ###
    { values: [1,1,1, 1,0,1, 1,0,1, 1,0,1, 1,1,1], target: [1,0,0,0,0] },
    // 1 :  .#.    .#.    .#.    .#.    .#.
    { values: [0,1,0, 0,1,0, 0,1,0, 0,1,0, 0,1,0], target: [0,1,0,0,0] },
    // 2 :  ###    ..#    ###    #..    ###
    { values: [1,1,1, 0,0,1, 1,1,1, 1,0,0, 1,1,1], target: [0,0,1,0,0] },
    // 3 :  ###    ..#    ###    ..#    ###
    { values: [1,1,1, 0,0,1, 1,1,1, 0,0,1, 1,1,1], target: [0,0,0,1,0] },
    // 4 :  #.#    #.#    ###    ..#    ..#
    { values: [1,0,1, 1,0,1, 1,1,1, 0,0,1, 0,0,1], target: [0,0,0,0,1] },
]

module.exports = datas;

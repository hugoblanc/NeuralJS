import { Layer } from './layer';
export class Network {

    public layers: Layer[] = [];

    constructor(...layersSizes: number[]) {
        let prevLayer;
        for (const nbNeurons of layersSizes) {
            prevLayer = new Layer(nbNeurons, prevLayer);
            this.layers.push(prevLayer);
        }
    }
}
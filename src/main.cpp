#include <iostream>
#include "Neural-Network/NeuralNetwork.hpp"
#include "Activation-Functions/ReLU.hpp"
#include "Activation-Functions/Softmax.hpp"
#include "Activation-Functions/Identity.hpp"
#include "Activation-Functions/Sigmoid.hpp"
#include "./aliases.hpp"
#include "./Loss-Functions/CategoricalCrossEntropy.hpp"
#include "./Loss-Functions/MeanSquaredError.hpp"

int main(){
    int numOfInputNodes{2};
    NN:: NeuralNetwork neuralNet{numOfInputNodes};
    neuralNet.addLayer(2,NN::Sigmoid{});

    int epochs{200};

    NN::matrix xtrain{
        {0,0},
        {0,1},
        {1,0},
        {100,99},
        {103,97},
        {103,104},
    };

    NN::row ytrain{0,0,0,1,1,1};
    
    neuralNet.train(xtrain, ytrain, epochs, 0.1, NN::CategoricalCrossEntropy{});

    // neuralNet.printWeights();
    // neuralNet.printBiases();

    double y { neuralNet.predict(NN::row{90, 101}) };
    std::cout << "x={90, 101}" << " y=" << y << '\n';

    return 0;
}
#include <iostream>
#include "Neural-Network/NeuralNetwork.hpp"
#include "Activation-Functions/ReLU.hpp"
#include "Activation-Functions/Softmax.hpp"
#include "./aliases.hpp"
#include "./Loss-Functions/CategoricalCrossEntropy.hpp"

int main(){
    int numOfInputNodes{2};
    NN:: NeuralNetwork neuralNet{numOfInputNodes};
    neuralNet.addLayer(8,NN::ReLU{});
    // neuralNet.addLayer(10,NN::ReLU{});
    neuralNet.addLayer(2,NN::Softmax{});
    int epochs{3};

    NN::matrix xtrain{
        {0,0},
        {0,1},
        {1,0},
        {10,10},
        {10,11},
        {11,10}
    };

    NN::row ytrain{0,0,0,1,1,1};
    
    neuralNet.train(xtrain, ytrain, epochs, 0.1, NN::CategoricalCrossEntropy{});

    return 0;
}
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
    int numOfInputNodes{1};
    NN:: NeuralNetwork neuralNet{numOfInputNodes};
    neuralNet.addLayer(1,NN::Identity{});
    int epochs{20};

    NN::matrix xtrain{
        {0},
        {1},
        {2},
    };

    NN::row ytrain{0,1,2};
    
    neuralNet.train(xtrain, ytrain, epochs, 0.1, NN::MeanSquaredError{});

    return 0;
}
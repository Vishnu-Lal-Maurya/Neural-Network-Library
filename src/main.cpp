#include <iostream>
#include "Neural-Network/NeuralNetwork.hpp"
#include "Activation-Functions/ReLU.hpp"
#include "Activation-Functions/Softmax.hpp"

int main(){
    int numOfInputNodes{5};
    NN:: NeuralNetwork neuralNet{numOfInputNodes};
    neuralNet.addLayer(10,NN::ReLU{});
    neuralNet.addLayer(10,NN::ReLU{});
    neuralNet.addLayer(5,NN::Softmax{});
    int epochs{3};
    NN::row input(numOfInputNodes,1.0);
    
    neuralNet.train(epochs, input);

    return 0;
}
#include <iostream>
#include "Neural-Network/NeuralNetwork.hpp"
#include "Activation-Functions/ReLU.hpp"
#include "Activation-Functions/Softmax.hpp"

int main(){
    int numOfInputNodes{10};
    NN:: NeuralNetwork neuralNet{numOfInputNodes};
    neuralNet.addLayer(20,NN::ReLU{});
    neuralNet.addLayer(20,NN::ReLU{});
    neuralNet.addLayer(20,NN::ReLU{});
    neuralNet.addLayer(5,NN::Softmax{});
    int epochs{1};
    NN::row input(10,1);
    
    // neuralNet.train(epochs, input);

    return 0;
}
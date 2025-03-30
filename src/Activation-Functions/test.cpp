#include "ReLU.hpp"
#include "Sigmoid.hpp"
#include "Softmax.hpp"
#include <iostream>

int main(){
    NN::Sigmoid sigmoid;
    NN::ReLU relu;

    NN::row input{1.0, 2.0, 0.0, -1.0};

    NN::row output{ sigmoid.activate(input) };

    NN::row activationOutput{ relu.activate(input) };
    NN::row derivativeOutput{ relu.derivate(input) };

    for(auto x: output){
        std::cout << x << ' ';
    }
    std::cout << '\n';

<<<<<<< HEAD
    NN::row derivatives{ sigmoid.derivate(input) };
    for(auto x: derivatives){
=======
    std::cout << "activation output = ";
    for(auto x: activationOutput){
        std::cout << x << ' ';
    }
    std::cout << '\n';
    std::cout << "derivate output = ";
    for(auto x: derivativeOutput){
>>>>>>> 0a6d9fa2193e5562f2db098c4bc64705cba0ad47
        std::cout << x << ' ';
    }
    std::cout << '\n';

<<<<<<< HEAD
    // using NN::operator<<;
    std::cout << derivatives << '\n';
=======
>>>>>>> 0a6d9fa2193e5562f2db098c4bc64705cba0ad47
}
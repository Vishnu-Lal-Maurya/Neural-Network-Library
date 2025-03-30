#include "ReLU.hpp"
#include "Sigmoid.hpp"
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

    std::cout << "activation output = ";
    for(auto x: activationOutput){
        std::cout << x << ' ';
    }
    std::cout << '\n';
    std::cout << "derivate output = ";
    for(auto x: derivativeOutput){
        std::cout << x << ' ';
    }
    std::cout << '\n';

}
#include "Tanh.hpp"
#include <iostream>

int main(){
    NN::Tanh tanh;

    NN::row input{1.0, 2.0, 0.0, -1.0};


    NN::row activationOutput{ tanh.activate(input) };
    NN::row derivativeOutput{ tanh.derivate(input) };



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
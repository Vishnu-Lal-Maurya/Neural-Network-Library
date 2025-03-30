#include "ReLU.hpp"
#include "Sigmoid.hpp"
#include <iostream>

int main(){
    NN::Sigmoid sigmoid;
    NN::row input{1.0, 2.0, 0.0, -1.0};
    NN::row output{ sigmoid.activate(input) };
    for(auto x: output){
        std::cout << x << ' ';
    }
    std::cout << '\n';
}
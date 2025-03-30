// #include "Tanh.hpp"
#include "CategoricalCrossEntropy.hpp"
#include <iostream>
#include "../aliases.hpp"

int main(){
     
    NN:: CategoricalCrossEntropy temp;
    // NN:: row v;
    

    NN::row input{0,1,0};
    NN:: row output{0.1,0.8,0.1};
    std::cout << temp.computeCost(input, output) << std::endl;
    


}
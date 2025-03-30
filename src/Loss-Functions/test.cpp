// #include "Tanh.hpp"
#include "CategoricalCrossEntropy.hpp"
#include <iostream>
#include "../aliases.hpp"
#include "../utils/operations.hpp"

int main(){
     
    NN:: matrix v1{{1,2},{2,3},{3,4}};
    // NN:: matrix v2{{1,2},{2,3}};
    NN:: matrix result{NN::transpose(v1)};
    for(int i{0}; i < result.size(); ++i){
        for(int j{0}; j < result[0].size(); ++j){
            std::cout << result[i][j] << ' ';
        }
        std::cout << std::endl;
        
    }






}
#include <iostream>
#include "Neural-Network/NeuralNetwork.hpp"
#include "Activation-Functions/ReLU.hpp"
#include "Activation-Functions/Softmax.hpp"
#include "Activation-Functions/Identity.hpp"
#include "Activation-Functions/Sigmoid.hpp"
#include "./aliases.hpp"
#include "./Loss-Functions/CategoricalCrossEntropy.hpp"
#include "./Loss-Functions/MeanSquaredError.hpp"
#include <fstream>
#include <string>
#include <sstream>

std::pair<NN::matrix, NN::row> readFile(std::string path ){
    std::cout << "Path: " << path << '\n';
    std::ifstream inputFile{ path };
    std::string str{};
    NN::matrix x{};
    NN::row y{};
    while (std::getline(inputFile, str))
    {
        std::stringstream os {str};
        std::string cell{};

        NN::row curr{};
        while (getline(os, cell, ',')) {
            curr.push_back(stod(cell));
        }

        y.push_back(curr.back());
        curr.pop_back();
        x.push_back(curr);
    }
    return {x,y};
}

int main(){

    std::pair<NN::matrix, NN::row> data {readFile("D://Neural-Network-Library//Simple-Datasets//LinearReg.csv")};
    NN::matrix xTrain { data.first };
    NN::row yTrain { data.second };


    using namespace NN;
    for(auto &row:xTrain){
        std::cout << row << '\n';
    }

    std::cout << yTrain << "\n";


    NN::NeuralNetwork neuralNet{1};
    // neuralNet.addLayer(5,NN::ReLU{});
    // neuralNet.addLayer(3,NN::ReLU{});
    neuralNet.addLayer(1,NN::Identity{});

    neuralNet.train(xTrain, yTrain, 100, 0.1, NN::MeanSquaredError{});
    NN::row output{ neuralNet.predict(xTrain) };
    // std::pair<NN::matrix, NN::row> testData {readFile("D://Neural-Network-Library//Simple-Datasets//LinearRegTest.csv")};
    // NN::matrix xTest { testData.first };
    // NN::row yTest { testData.second };    
    // using namespace NN;
    // for(auto r: xTest){
    //     std::cout << r << '\n';
    // }
    // // std::cout << yTest << '\n';

    // NN::row output { neuralNet.predict(xTest) };
    std::cout << output << '\n';

    return 0;
}
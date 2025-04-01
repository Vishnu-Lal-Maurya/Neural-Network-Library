#define DEBUG d

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

    // std::pair<NN::matrix, NN::row> data {readFile("D://Neural-Network-Library//Simple-Datasets//LinearReg.csv")};
    // NN::matrix xTrain { data.first };
    // NN::row yTrain { data.second };


    // NN::NeuralNetwork neuralNet{1};
    // NN::Identity idt{};
    // neuralNet.addLayer(1,idt);

    // NN::MeanSquaredError mse{};
    // neuralNet.train(xTrain, yTrain, 300, 0.001, mse);

    // std::pair<NN::matrix, NN::row> testData{readFile("D://Neural-Network-Library//Simple-Datasets//LinearRegTest.csv")};
    // NN::matrix xTest { testData.first };
    // NN::row yTest { testData.second };
    // NN::row output{ neuralNet.predict(xTest) };
    // using namespace NN;
    // std::cout << output << '\n';

    NN::matrix x{
        {0,0},
        {1,1},
        {2,2},
        {0,10},
        {1,11},
        {2,12}
    };
    NN::row y{0,0,0,1,1,1}; 

    NN::NeuralNetwork nn{2};
    NN::ReLU relu{};
    NN::Softmax sft{};
    nn.addLayer(1,relu);
    nn.addLayer(2,sft);

    NN::CategoricalCrossEntropy los{};
    nn.train(x,y,4,0.1,los);

    double pred = nn.predict(NN::row{1,0});
    std::cout << pred << '\n';
    return 0;
}
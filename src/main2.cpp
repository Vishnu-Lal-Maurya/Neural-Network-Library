#define DEBUG1
#include <iostream>
#include "Neural-Network/NeuralNetwork.hpp"
#include "Activation-Functions/ReLU.hpp"
#include "Activation-Functions/Softmax.hpp"
#include "Activation-Functions/Identity.hpp"
#include "Activation-Functions/Sigmoid.hpp"
#include "./aliases.hpp"
#include "./Loss-Functions/CategoricalCrossEntropy.hpp"
#include "./Loss-Functions/MeanSquaredError.hpp"
#include "Activation-Functions/Tanh.hpp"
#include <fstream>
#include <string>
#include <sstream>
#include <random>

using NN::row;
using NN::matrix;
typedef std::pair<matrix, row> DataPair;

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

DataPair normalizeData(const matrix& x, const row& y) {
    matrix xNorm = x;
    row yNorm = y;
    
    // Find min and max for each feature in x
    size_t numFeatures = x[0].size();
    row xMin(numFeatures, std::numeric_limits<double>::max());
    row xMax(numFeatures, std::numeric_limits<double>::lowest());
    double yMin = *std::min_element(y.begin(), y.end());
    double yMax = *std::max_element(y.begin(), y.end());
    
    for (const auto& row : x) {
        for (size_t i = 0; i < numFeatures; ++i) {
            xMin[i] = std::min(xMin[i], row[i]);
            xMax[i] = std::max(xMax[i], row[i]);
        }
    }
    
    // Normalize x
    for (auto& row : xNorm) {
        for (size_t i = 0; i < numFeatures; ++i) {
            row[i] = (row[i] - xMin[i]) / (xMax[i] - xMin[i]);
        }
    }
    
    // // Normalize y
    // for (size_t i = 0; i < yNorm.size(); ++i) {
    //     yNorm[i] = (yNorm[i] - yMin) / (yMax - yMin);
    // }
    
    return {xNorm, yNorm};
}

std::pair<matrix, row> zScoreNormalize(const matrix& x, const row& y) {
    matrix xNorm = x;
    row yNorm = y;

    size_t numFeatures = x[0].size();
    row xMean(numFeatures, 0.0);
    row xStd(numFeatures, 0.0);
    double yMean = 0.0;
    double yStd = 0.0;

    // Compute mean for each feature in x
    for (const auto& row : x) {
        for (size_t i = 0; i < numFeatures; ++i) {
            xMean[i] += row[i];
        }
    }

    for (size_t i = 0; i < numFeatures; ++i) {
        xMean[i] /= x.size();
    }

    // Compute standard deviation for each feature in x
    for (const auto& row : x) {
        for (size_t i = 0; i < numFeatures; ++i) {
            xStd[i] += std::pow(row[i] - xMean[i], 2);
        }
    }
    for (size_t i = 0; i < numFeatures; ++i) {
        xStd[i] = std::sqrt(xStd[i] / x.size());
        if (xStd[i] == 0) xStd[i] = 1;  // Prevent division by zero
    }

    // Normalize x
    for (auto& row : xNorm) {
        for (size_t i = 0; i < numFeatures; ++i) {
            row[i] = (row[i] - xMean[i]) / xStd[i];
        }
    }

    // // Compute mean and std for y
    // for (double val : y) {
    //     yMean += val;
    // }
    // yMean /= y.size();

    // for (double val : y) {
    //     yStd += std::pow(val - yMean, 2);
    // }
    // yStd = std::sqrt(yStd / y.size());
    // if (yStd == 0) yStd = 1;  // Prevent division by zero

    // // Normalize y
    // for (size_t i = 0; i < yNorm.size(); ++i) {
    //     yNorm[i] = (yNorm[i] - yMean) / yStd;
    // }
    // // Normalize y
    // for (size_t i = 0; i < yNorm.size(); ++i) {
    //     yNorm[i] = (yNorm[i] - yMean) / yStd;
    // }

    return {xNorm, y};
}

int main(){

    std::pair<NN::matrix, NN::row> data {readFile("/home/fireheart17/2025/NeutalNetProject/Neural-Network-Library/Simple-Datasets/Iris.csv")};
    // #ifdef piyush
    NN::matrix xTrain { data.first };
    NN::row yTrain { data.second };
    std::pair<NN::matrix, NN::row> normalizedData = zScoreNormalize(xTrain, yTrain);
    // std::pair<NN::matrix, NN::row> normalizedData = data;
    NN::matrix xTrainNorm = normalizedData.first;
    NN::row yTrainNorm = normalizedData.second;

    // NN::matrix xTrainNorm = xTrain;
    // NN::row yTrainNorm = yTrain;
    // NN::NeuralNetwork neuralNet{1};
    // NN::Identity idt{};
    // neuralNet.addLayer(1,idt);

    // NN::MeanSquaredError mse{};
    // neuralNet.train(xTrain, yTrain, 300, 0.001, mse);


    // NN::matrix x{
    //     {0},
    //     {1},
    //     {2},
    //     {3},
    //     {4},
    //     {5}
    // };
    // NN::row y{0,1,2,3,4,5}; 

    NN::NeuralNetwork nn{4};
    NN::ReLU relu{};
    NN::Softmax sft{};
    NN::Sigmoid sgd{};
    NN::Identity idt{};
    NN::Tanh tnh{};
    nn.addLayer(2,relu);
    nn.addLayer(5,relu);
    nn.addLayer(3,sft);

    NN::CategoricalCrossEntropy los{};
    nn.train(xTrainNorm, yTrainNorm, 2000, 0.001, los);

    std::cout << "\n\nTesting Started.................\n";

    std::pair<NN::matrix, NN::row> testData{readFile("/home/fireheart17/2025/NeutalNetProject/Neural-Network-Library/Simple-Datasets/IrisTest.csv")};
    NN::matrix xTest { testData.first };
    NN::row yTest { testData.second };

    std::pair<NN::matrix, NN::row> normalizedDataTest = zScoreNormalize(xTest, yTest);
    NN::matrix xTestNorm = normalizedDataTest.first;
    NN::row yTestNorm = normalizedDataTest.second;
    NN::row output{ nn.predict(xTestNorm) };

    using namespace NN;
    std::cout << output << '\n';

    // double pred = nn.predict(NN::row{7});
    // std::cout << pred << '\n';

    // double pred1 = nn.predict(NN::row{10});
    // std::cout << pred1 << '\n';
    // #endif
    // dout.close();
    
    return 0;
}
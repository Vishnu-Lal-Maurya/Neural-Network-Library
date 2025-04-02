
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
#include "Metrics/Classification-Metrics.hpp"
#include "Metrics/Regression-Metrics.hpp"

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

    // Compute mean and std for y
    for (double val : y) {
        yMean += val;
    }
    yMean /= y.size();

    for (double val : y) {
        yStd += std::pow(val - yMean, 2);
    }
    yStd = std::sqrt(yStd / y.size());
    if (yStd == 0) yStd = 1;  // Prevent division by zero

    // Normalize y
    for (size_t i = 0; i < yNorm.size(); ++i) {
        yNorm[i] = (yNorm[i] - yMean) / yStd;
    }

    return {xNorm, yNorm};
}

int main(){

    std::pair<NN::matrix, NN::row> data {readFile("/home/vishnu/Desktop/Project/Neural-Network-Library/Simple-Datasets/Iris.csv")};
    NN::matrix xTrain { data.first };
    NN::row yTrain { data.second };
    std::pair<NN::matrix, NN::row> normalizedData = normalizeData(xTrain, yTrain);
    NN::matrix xTrainNorm = normalizedData.first;
    NN::row yTrainNorm = normalizedData.second;
    // using namespace NN;
    // std::cout << "xTrain\n";
    // std::cout << xTrain << '\n';

    // std::cout << "yTrain\n";
    // std::cout << yTrain << '\n';
   

    NN::NeuralNetwork nn{4};
    NN::ReLU relu{};
    NN::Softmax sft{};
    NN::Sigmoid sgd{};
    NN::Identity idt{};
    NN::Tanh tnh{};
    nn.addLayer(4,sgd);
    nn.addLayer(3,sft);
    // nn.addLayer(1,idt);

    NN::CategoricalCrossEntropy los{};
    nn.train(xTrainNorm, yTrainNorm, 2000, 0.01, los);
   

    data = readFile("/home/vishnu/Desktop/Project/Neural-Network-Library/Simple-Datasets/Iris_test.csv");
    NN::matrix xTest { data.first };
    NN::row yTest { data.second };
    normalizedData = normalizeData(xTest, yTest);
    NN::matrix xTestNorm = normalizedData.first;
    NN::row yTestNorm = normalizedData.second;

    NN::row yPredict{nn.predict(xTestNorm)};

    using namespace NN;
    std::cout << yTest << '\n';
    std::cout << yPredict << '\n';

    NN::ClassificationMetrics clme{yTestNorm,yPredict};

    std::cout << clme.getAccuracy() << '\n';
    
    return 0;
}
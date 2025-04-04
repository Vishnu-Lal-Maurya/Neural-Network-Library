#ifndef LOADMODEL_H
#define LOADMODEL_H

#include "../Neural-Network/NeuralNetwork.hpp"
#include <nlohmann/json.hpp>
#include <filesystem>
#include <fstream>
#include "../aliases.hpp"
namespace fs = std::filesystem;
using json = nlohmann::json;

namespace NN{
    NeuralNetwork loadModel(std::string_view filePath){
        const fs::path jsonFilePath { filePath };
        if(jsonFilePath.extension() != ".json"){
            throw "File type not supported";
        } 
        // Create a json object from file stream
        std::ifstream myFileStream { jsonFilePath.c_str() };
        json j = json::parse(myFileStream);
    
        // Extract the number of input nodes
        int inputDim{ j["inputDim"] };
        NeuralNetwork nn{ inputDim };

        int numLayers { j["numberOfLayers"] };
        int prevOutput { inputDim };
        for(int i{0}; i<numLayers; ++i){
            // Extract the saved properties of the layer --
            int inputSize  { j["Layers"][i]["inputSize"] };
            int outputSize { j["Layers"][i]["outputSize"] };
            double dropOut { j["Layers"][i]["dropOut"] };
            int activationFunctionEnum { j["Layers"][i]["activationFunction"] };
            row bias = j["Layers"][i]["bias"];
            matrix weights = j["Layers"][i]["weights"];

            // check for valid input 
            assert((inputSize == prevOutput) && "Layer structure not valid");
            assert((outputSize == static_cast<int>(bias.size())) && "Layer structure not valid"); 
            assert((outputSize == weights.size() && outputSize && (inputSize == weights[0].size()) && "Layer structure not valid")); 
            prevOutput = outputSize;

            nn.addLayer(inputSize, outputSize, weights, bias, activationFunctionEnum, dropOut);

            #ifdef DEBUG_LOAD
            std::cout << bias << '\n';
            std::cout << weights << '\n';
            std::cout << inputSize << ' ' << outputSize << ' ' << dropOut << '\n';
            #endif
        }
        return nn;
    }
}

#endif
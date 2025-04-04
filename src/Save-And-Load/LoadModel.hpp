#ifndef LOADMODEL_H
#define LOADMODEL_H

#include "../Neural-Network/NeuralNetwork.hpp"
#include <nlohmann/json.hpp>
#include <filesystem>
namespace fs = std::filesystem;

namespace NN{
    NeuralNetwork loadModel(std::string_view filePath){
        const fs::path jsonFilePath { filePath };
        if(jsonFilePath.extension() != ".json"){
            throw "File type not supported";
        } 
        NeuralNetwork nn{5};
        return nn;
    }
}

#endif
#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include "../aliases.h"
#include "../utils/operations.hpp"
#include "../Layers/Layer.hpp"
#include "../Activation-Functions/ActivationFunction.hpp"
#include "../Loss-Functions/LossFunction.hpp"

class NeuralNetwork{
public:
   NeuralNetwork() = default;
   void addLayer(int numberOfNodes, NN::ActivationFunction& activationFunction){

   }  
   
private: 
   std::vector<NN::Layer> m_layers{};
   NN::LossFunction& m_lossFunction;

};

#endif
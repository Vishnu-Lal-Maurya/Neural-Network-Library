#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include "../aliases.h"
#include "../utils/operations.hpp"
#include "../Layers/Layer.hpp"
#include "../Activation-Functions/ActivationFunction.hpp"
#include "../Loss-Functions/LossFunction.hpp"

namespace NN{

   class NeuralNetwork{

   public:
      NeuralNetwork(int inputDim)
      : m_inputDim{inputDim}
      {

      }
      
      void addLayer(int currNumOfNodes, const NN::ActivationFunction& activationFunction){
         int prevNumOfNodes{m_inputDim};
         if(m_layers.size()){
            prevNumOfNodes = m_layers.back().getOutputSize();
         }
         m_layers.push_back(NN::Layer{prevNumOfNodes,currNumOfNodes,activationFunction});
      }  

   private:
      std::vector<NN::Layer> m_layers{};
      int m_inputDim{};

      void forward(row input){

         for(auto layer : m_layers){
            input = layer.forwardPropagate(input);
         }

      }

   };

};

#endif
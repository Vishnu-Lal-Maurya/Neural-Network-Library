#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include "../aliases.hpp"
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

      void train(matrix& xTrain, row& yTrain, int epochs, double learningRate, const LossFunction& lossFunction){
         assert((xTrain.size() == yTrain.size()) && "x_train and y_train sizes don't match");

         
         // Run the epochs -
         for(int ep{1}; ep<=epochs; ++ep){
            double totalLoss{ 0.0 };
            for(int i{0}; i<xTrain.size(); ++i){

               row output { forward(xTrain[i]) };
               
               // Doing one hot encoding !(if last layer has 1 output node)'
               row actualOutput(m_layers.back().getOutputSize(),0.0);
               if(m_layers.back().getOutputSize()!=1){
                  int index{ static_cast<int>(yTrain[toUZ(i)]) };
                  actualOutput[toUZ(index)] = 1.0;
               }  
               else{
                  actualOutput[0] = yTrain[toUZ(i)];
               }
               totalLoss += lossFunction.computeCost(actualOutput, output);

               // Calculate derivative of the loss function 
               row dLoss = lossFunction.derivative(actualOutput, output);

               // @todo 
               // implement backward propagation
               backward(dLoss, learningRate);
            }

            double avgLoss { totalLoss/(static_cast<double>(yTrain.size())) };
            std::cout << "Average loss after epoch " << ep << ": " << avgLoss << '\n';
            
         }
      }
      
   private:
      
      std::vector<NN::Layer> m_layers{};
      int m_inputDim{};
      
      row forward(row input){
   
         for(auto& layer : m_layers){
            input = layer.forwardPropagate(input);
         }

         return std::move_if_noexcept(input);
      }

      void backward(const row& dLoss, double learningRate){
         row prev{ dLoss };
         using namespace NN;
         // std::cout << "dActivated: ";
         // std::cout << prev << '\n';
         for(auto it{ m_layers.rbegin() }; it!=m_layers.rend(); ++it){
            // std::cout << "I'm in the loop: \n";
            prev = it->backwardPropogate(prev, learningRate);
            // std::cout << prev << '\n';
         }
      }
   };

};

#endif
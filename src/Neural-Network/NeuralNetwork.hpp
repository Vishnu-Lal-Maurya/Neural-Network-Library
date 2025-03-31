#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include "../aliases.hpp"
#include "../utils/operations.hpp"
#include "../Layers/Layer.hpp"
#include "../Activation-Functions/ActivationFunction.hpp"
#include "../Loss-Functions/LossFunction.hpp"
#include <algorithm>

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

      row predict(const matrix& x);

      double predict(const row& x);

      void train(const matrix& xTrain, const row& yTrain, int epochs, double learningRate, const LossFunction& lossFunction){
         assert((xTrain.size() == yTrain.size()) && "x_train and y_train sizes don't match");

         
         // Run the epochs -
         for(int ep{1}; ep<=epochs; ++ep){
            double totalLoss{ 0.0 };
            for(int i{0}; i<xTrain.size(); ++i){

               row output { forward(xTrain[i]) };

               row actualOutput(m_layers.back().getOutputSize(),0.0);
               if(isClassification()){
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

      void printWeights(){
         std::cout << "Weights: \n";
         for(auto r: m_layers.back().getWeights()){
            std::cout << r << '\n';
         }
      }

      void printBiases(){
         std::cout << "Biases: \n";
         std::cout <<  m_layers.back().getBiases() << '\n';
      }
      
   private:
      
      std::vector<NN::Layer> m_layers{};
      int m_inputDim{};
      
      bool isClassification(){
         return (m_layers.back().getOutputSize() != 1);
      }

      row forward(row input){
         for(auto& layer : m_layers){
            input = layer.forwardPropagate(input);
         }
         return input;
      }

      void backward(const row& dLoss, double learningRate){
         row prev{ dLoss };
         for(auto it{ m_layers.rbegin() }; it!=m_layers.rend(); ++it){
            prev = it->backwardPropogate(prev, learningRate);
         }
      }
   };

   inline row NeuralNetwork::predict(const matrix& x){
      row result{};
      for(auto input: x){
         result.push_back(predict(input));
      }
      return result;
   }

   inline double NeuralNetwork::predict(const row& x){
      row output { forward(x) };
      std::cout << output << '\n';
      if(isClassification()){
         double predictedClass{ static_cast<double>(max_element(output.begin(), output.end()) - output.begin())};
         return predictedClass;
      }
      else{
         return output[0];
      }
   }

};

#endif
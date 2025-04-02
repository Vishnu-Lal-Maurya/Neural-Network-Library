#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include "../aliases.hpp"
#include "../utils/RandomOperations.hpp"
#include "../utils/Operations.hpp"
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
      
      void addLayer(int currNumOfNodes, const NN::ActivationFunction& activationFunction, double dropout = 0.0){
         int prevNumOfNodes{m_inputDim};
         if(m_layers.size()){
            prevNumOfNodes = m_layers.back().getOutputSize();
         }
         m_layers.push_back(NN::Layer{prevNumOfNodes,currNumOfNodes,activationFunction, dropout});
      }  

      row predict(const matrix& x);

      double predict(const row& x);

      void train(matrix xTrain, row yTrain, int epochs, double learningRate0, const LossFunction& lossFunction, double decayRate = 0.0, int timeInterval = 1);

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
      
      bool isClassification() const {
         return (m_layers.back().getOutputSize() != 1);
      }

      row forward(row input, bool toDrop = false){
         #ifdef DEBUG
            std::cout << "Forward Propagation started...\n";
         #endif
         for(auto& layer : m_layers){
            input = layer.forwardPropagate(input, toDrop);
         }
         #ifdef DEBUG
            std::cout << "Result of forward Propagation:";
            std::cout << input << '\n';
         #endif
         return input;
      }

      void backward(const row& dLoss, double learningRate){
         #ifdef DEBUG
            std::cout << "Backward Propagation started...\n";
         #endif
         row prev{ dLoss };
         for(auto it{ m_layers.rbegin() }; it!=m_layers.rend(); ++it){
            prev = it->backwardPropagate(prev, learningRate);
         }
         #ifdef DEBUG
            std::cout << "Weights and biases after backProp:\n";
            for(const auto& layer: m_layers){
               std::cout << "Weights:\n";
               std::cout << layer.getWeights() << '\n';
               std::cout << "Biases:\n";
               std::cout << layer.getBiases() << '\n';
            }
         #endif
      }


   };

   inline row NeuralNetwork::predict(const matrix& x){
      row result{};
      result.reserve(x.size());
      for(auto input: x){
         result.push_back(predict(input));
      }
      return result;
   }

   inline double NeuralNetwork::predict(const row& x){
      row output { forward(x) };
      if(isClassification()){
         double predictedClass{ static_cast<double>(max_element(output.begin(), output.end()) - output.begin())};
         return predictedClass;
      }
      else{
         return output[0];
      }
   }


   inline void NeuralNetwork::train(matrix xTrain, row yTrain, int epochs, double learningRate0, const LossFunction& lossFunction, double decayRate = 0.0, int timeInterval = 1){
      assert((xTrain.size() == yTrain.size()) && "x_train and y_train sizes don't match");
      assert(timeInterval > 0 && "timeInterval for learning rate decay should be positive");
      assert(decayRate >= 0.0 && "decayRate should be non negative");
      
      // Run the epochs -
      for(int ep{1}; ep<=epochs; ++ep){
         shuffleData(xTrain, yTrain);
         #ifdef DEBUG
            std::cout << "\n\nEpoch: " << ep << "............................\n";
         #endif
         double totalLoss{ 0.0 };
         // applying learing rate decay
         double learningRate{learningRate0 / (1.0 + decayRate * ((ep-1) / timeInterval))};
         #ifdef DEBUGLR
         std::cout << "learning Rate: "<<learningRate << std::endl;
         #endif

         for(int i{0}; i<xTrain.size(); ++i){

            row output { forward(xTrain[i], true) };

            row actualOutput(m_layers.back().getOutputSize(),0.0);
            if(isClassification()){
               int index{ static_cast<int>(yTrain[toUZ(i)]) };
               actualOutput[toUZ(index)] = 1.0;
            }  
            else{
               actualOutput[0] = yTrain[toUZ(i)];
            }
            #ifdef DEBUG
               std::cout << "Loss: " << lossFunction.computeCost(actualOutput, output) << '\n';
            #endif
            double currentLoss{ lossFunction.computeCost(actualOutput, output) };
            assert((currentLoss>0.0) && "Loss can't be negative\n");
            totalLoss += currentLoss;

            // Calculate derivative of the loss function 
            row dLoss = lossFunction.derivative(actualOutput, output);
            #ifdef DEBUG
               std::cout << "dLoss: " << dLoss << '\n';
            #endif
            // @todo 
            // implement backward propagation
            backward(dLoss, learningRate);

         #ifdef DEBUG
            std::cout << "Result of forward Propagation:";
            std::cout << "\n\n";
         #endif
         }

         double avgLoss { totalLoss/(static_cast<double>(yTrain.size())) };
         std::cout << "Average loss after epoch " << ep << ": " << avgLoss << '\n';
         
      }
   }


};

#endif
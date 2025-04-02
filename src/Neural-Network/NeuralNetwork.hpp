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

      void train(matrix xTrain, row yTrain, int epochs, double learningRate, const LossFunction& lossFunction){
         assert((xTrain.size() == yTrain.size()) && "x_train and y_train sizes don't match");

         
         // Run the epochs -
         for(int ep{1}; ep<=epochs; ++ep){
            shuffleData(xTrain, yTrain);
            #ifdef DEBUG
               std::cout << "\n\nEpoch: " << ep << "............................\n";
            #endif
            double totalLoss{ 0.0 };
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
            for(auto layer: m_layers){
               std::cout << "Weights:\n";
               std::cout << layer.getWeights() << '\n';
               std::cout << "Biases:\n";
               std::cout << layer.getBiases() << '\n';
            }
         #endif
      }

      void shuffleData(NN::matrix& x, NN::row& y) {
         std::vector<size_t> indices(x.size());
         std::iota(indices.begin(), indices.end(), 0); // Fill with 0,1,2,...,N-1
     
         std::random_device rd;
         std::mt19937 g(rd()); // Random engine
     
         std::shuffle(indices.begin(), indices.end(), g); // Shuffle indices
     
         // Rearrange x and y according to shuffled indices
         NN::matrix xShuffled(x.size());
         NN::row yShuffled(y.size());
     
         for (size_t i = 0; i < indices.size(); ++i) {
             xShuffled[i] = x[indices[i]];
             yShuffled[i] = y[indices[i]];
         }
     
         x = std::move(xShuffled);
         y = std::move(yShuffled);
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
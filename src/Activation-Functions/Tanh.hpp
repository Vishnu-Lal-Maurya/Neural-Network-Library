#ifndef TANH_H 
#define TANH_H
#include <cmath>
#include "ActivationFunction.hpp"


namespace NN {

   class Tanh final: public ActivationFunction
   {
   public:
      row activate(const row& input) const override {
         row result(input);
         for(auto &ele : result){
            ele = std::tanh(ele);
         }
         return result;
      };

      row derivate(const row& input) const override {
         row result(activate(input));
         for(auto &ele : result){
            ele = 1 - ele*ele;
         }
         return result;
      }

      std::unique_ptr<ActivationFunction> clone() const override{
         return std::make_unique<Tanh>();
      }

      virtual ~Tanh() = default;
   };
   
 

}


#endif
#ifndef TANH_H 
#define TANH_H
#include <cmath>
#include "ActivationFunction.hpp"


namespace NN {

   class Tanh final: public ActivationFunction
   {
      public:
      row activate(const row& input) override {
         row result(input);
         for(auto &ele : result){
            ele = ( exp(ele) - exp(-ele) ) / ( exp(ele) + exp(-ele) );
         }
         return result;
      };

      row derivate(const row& input) override {
         row result(input);
         for(auto &ele : result){
            ele = 1 - ele*ele;
         }
         return result;
      }

   };
   
 

}


#endif TANH_H
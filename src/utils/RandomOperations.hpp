#ifndef RANDOMOPERATIONS_H
#define RANDOMOPERATIONS_H

#include "../aliases.hpp"
#include <algorithm>
#include <cassert>
#include <chrono>
#include <random>


namespace NN
{
  

   std::mt19937 rng(std::chrono::steady_clock::now().time_since_epoch().count());

   double randInRange(double lowerBound, double upperBound){
      return std::uniform_real_distribution<double> (lowerBound, upperBound)(rng);
   }

   row randRow(int cols){
      row result(cols);
      for(auto& i : result){
            i = randInRange(-0.1, 0.1);
      }
      return result;
   }

   matrix randMatrix(int rows, int cols){
      matrix result(rows);
      for(auto& currRow : result){
          currRow = randRow(cols);
      }
      return result;
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

} 


#endif
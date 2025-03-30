#ifndef RELU_H
#define RELU_H

#include "ActivationFunction.hpp"
#include <cassert>
#include <algorithm>

class ReLU final : public ActivationFunction{
public:
    MyMatrix<long double> activate(MyMatrix<long double>& input) override{
        assert(input.getCols()!=1 && "Only column vectors are allowed in activation function\n");
        int numRows { input.getRows() };
        MyMatrix<int> a(1,2);
        MyMatrix<long double> result(numRows, 1);
    }
};


#endif

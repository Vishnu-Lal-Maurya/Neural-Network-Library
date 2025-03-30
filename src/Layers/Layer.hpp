#ifndef LAYER_H
#define LAYER_H

#include "../aliases.hpp"
#include "../Activation-Functions/ActivationFunction.hpp"
#include "../utils/operations.hpp"

namespace NN
{

    class Layer
    {
    public:
        Layer(int m_outputSize, int m_inputSize, matrix m_w, row m_input, row m_bias, ActivationFunction m_activationFunction)
        {
            this->m_outputSize = m_outputSize;
            this->m_inputSize = m_inputSize;
            this->m_w = m_w;
            this->m_input = m_input;
            this->m_bias = m_bias;
            this->m_activationFunction = m_activationFunction;
        }

        row forwardPropagate(const row& input){

            row result(m_outputSize);

            for(int i{0}; i<m_outputSize; ++i){
                std::size_t x { toUZ(i) };
                result[i] = elementWiseRowOperations(input, m_w[x], '*') + m_bias[x];
            }

            // activate here..

            return result;
        }

    private:
        int m_outputSize{};
        int m_inputSize{};
        // w[i][j] denotes weight of the edge connecting ith neuron in the currLayer and the jth neuron in the prevLayer
        matrix m_w{};
        row m_input{};
        row m_bias{};
        row m_output{};
        ActivationFunction& m_activationFunction{};
    };

};

#endif
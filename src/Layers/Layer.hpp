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

        Layer(int inputSize, int outputSize, const ActivationFunction& activationFunction)
        : m_inputSize{ inputSize }
        , m_outputSize{ outputSize }
        , m_activationFunction{ activationFunction }
        , m_bias(outputSize)
        , m_weights(outputSize, row(inputSize))
        {
            // @todo -- 
            // We'll like to do random initialization here for weights in future
        }

        row forwardPropagate(const row& input){
            row result(m_outputSize);

            for(int i{0}; i<m_outputSize; ++i){
                std::size_t idx { toUZ(i) };
                result[idx] = dot(input, m_weights[idx]) + m_bias[idx];
            }
            
            // Activate the output
            result = m_activationFunction.activate(result);

            return std::move_if_noexcept(result);
        }

        int getOutputSize() const { return m_outputSize; }

        int getInputSize() const { return m_inputSize; }

    private:
        int m_outputSize{};
        int m_inputSize{};
        // w[i][j] denotes weight of the edge connecting ith neuron in the currLayer and the jth neuron in the prevLayer
        matrix m_weights{};
        row m_input{};
        row m_bias{};
        const ActivationFunction& m_activationFunction;
    };

};

#endif
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
            m_computed = result;

            // Activate the output
            result = m_activationFunction.activate(result);
            m_ativated = result;
            return std::move_if_noexcept(result);
        }

        row backwardPropogate(const row& dActivatedCurr, const row& activatedPrev, const double learningRate){

            matrix m_dweights{};
            row m_dbias{};
            row m_dcomputed{};

            m_dcomputed = NN::mul(dActivatedCurr,m_activationFunction.derivate(m_computed));
            m_dweights = NN::matMul(m_dcomputed,activatedPrev); 
            m_dbias = m_dcomputed;

            row dActivatedPrev = NN::matToRow(NN::matMul(NN::transpose(m_weights),NN::rowToColMatrix(m_dcomputed)));
           
            m_weights = m_weights - NN::mul(learningRate,m_dweights);
            m_bias = m_bias - NN::mul(learningRate,m_dbias);

            return dActivatedPrev;
            
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
        row m_computed{};
        row m_ativated{};


        const ActivationFunction& m_activationFunction;
    };

};

#endif
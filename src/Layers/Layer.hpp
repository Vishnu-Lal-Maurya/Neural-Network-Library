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
        , m_bias(outputSize,0.0)
        , m_weights(outputSize, row(inputSize,0.0))
        {
            // @todo -- 
            // We'll like to do random initialization here for weights in future
        }

        row forwardPropagate(const row& input){
            // store the input for backProp
            m_input = input;
            // row result(m_inputSize,0); 
            // for(int i{0}; i<m_outputSize; ++i){
            //     std::size_t idx { toUZ(i) };
            //     result[idx] = dot(input, m_weights[idx]) + m_bias[idx];
            // }

            // m_computed =  w@input + rowToCol(b)
            // Store it for use in backProp (Don't remove it)
            m_computed = NN::matToRow(NN::add(NN::matMul(m_weights,NN::rowToColMatrix(input)),NN::rowToColMatrix(m_bias)));

            // Activate the output
            // m_computed = result;
            
            row result = m_activationFunction.activate(m_computed);
            return std::move_if_noexcept(result);
        }

        row backwardPropogate(const row& dActivatedCurr, double learningRate){

            row dcomputed = NN::mul(dActivatedCurr,m_activationFunction.derivate(m_computed));
            matrix dweights = NN::matMul(NN::rowToColMatrix(dcomputed),m_input); 
            row dbias = dcomputed;
            
            
            row dActivatedPrev = NN::matToRow(NN::matMul(NN::transpose(m_weights),NN::rowToColMatrix(dcomputed)));
            m_weights = NN::sub(m_weights, NN::mul(learningRate,dweights));
            m_bias = NN::sub(m_bias, NN::mul(learningRate,dbias));

            return dActivatedPrev;
        }


        int getOutputSize() const { return m_outputSize; }

        int getInputSize() const { return m_inputSize; }

        matrix getWeights() const { return m_weights; }

        row getBiases() const { return m_bias; }

    private:
        int m_outputSize{};
        int m_inputSize{};
        // w[i][j] denotes weight of the edge connecting ith neuron in the currLayer and the jth neuron in the prevLayer
        matrix m_weights{};
        row m_input{};
        row m_bias{};
        row m_computed{};


        const ActivationFunction& m_activationFunction;
    };

};

#endif
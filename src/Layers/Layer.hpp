#ifndef LAYER_H
#define LAYER_H

#include "../aliases.hpp"
#include "../Activation-Functions/ActivationFunction.hpp"
#include "../utils/Operations.hpp"

namespace NN
{
    class Layer
    {
    public:

        Layer(int inputSize, int outputSize, const ActivationFunction& activationFunction, double dropout)
        : m_inputSize{ inputSize }
        , m_outputSize{ outputSize }
        , m_activationFunction{ activationFunction }
        , m_bias(outputSize,0.0)
        , m_weights{ randMatrix(outputSize, inputSize) }
        , m_dropout{ dropout }
        {
            // @todo -- 
            // We'll like to do random initialization here for weights in future

        }

        row forwardPropagate(const row& input, bool toDrop = false){
            // store the input for backProp
            m_input = input;

            // m_computed =  w@input + rowToCol(b)
            // Store it for use in backProp (Don't remove it)
            m_computed = NN::matToRow(NN::add(NN::matMul(m_weights,NN::rowToColMatrix(input)),NN::rowToColMatrix(m_bias)));

            // Activate the output
            // m_computed = result;            
            row result = m_activationFunction.activate(m_computed);

            m_dropVector.assign(result.size(),0.0);
            
            for(auto& i: m_dropVector){
                if(!(toDrop && randInRange(0.0,1.0) <= m_dropout)){
                    i = 1.0;
                }
            }

            #ifdef DEBUG1
            std::cout << "Dropout Vector while Forward Prop\n";
            for(auto i: m_dropVector){
                std::cout << i << ' ';
            }
            std::cout << std::endl;
            std::cout << std::endl;
            #endif

            // applying dropout
            result = mul(result, m_dropVector);
            // scaling so that expected value of the output remains same
            result = mul(result, 1.0 / (1.0 - m_dropout));

            #ifdef DEBUG
                std::cout << "Result of forward prop in layer: ";
                std::cout << result << '\n';
            #endif
            return result;
        }

        row backwardPropagate(const row& dActivatedCurr, double learningRate){

            row dcomputed = NN::mul(dActivatedCurr,m_activationFunction.derivate(m_computed));

            #ifdef DEBUG1
            std::cout << "Dropout Vector while backward Prop\n";
            for(auto i: m_dropVector){
                std::cout << i << ' ';
            }
            std::cout << std::endl;
            std::cout << std::endl;
            #endif
            // applying dropout
            dcomputed = mul(dcomputed, m_dropVector);

            // scaling so that expected value of the output remains same
            dcomputed = mul(dcomputed, 1.0 / (1.0 - m_dropout));

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
        row m_dropVector{};
        double m_dropout{};


        const ActivationFunction& m_activationFunction;
    };

};

#endif
/*!
 * \file
 *
 * A customised network of FeedForwardConns.
 *
 * \author Seb James
 * \date March 2022
 */
#pragma once

#include "NetConn.h"
#include <morph/vVector.h>
#include <utility>
#include <iostream>
#include <list>
#include <vector>
#include <sstream>
#include <ostream>
#include <map>

namespace morph {
    namespace nn {

        /*!
         * A feedforward network class which holds a runtime-selectable set of neuron
         * layers and the connections between the layers. Note that in this class, the
         * connections are always between adjacent layers; from layer l to layer l+1.
         */
        template <typename T>
        struct SpecialNet // LeakyIntegratorNet?
        {
            //! Constructor takes a vector specifying the number of neurons in each
            //! layer (\a layer_spec) AND a specification for the connections. Or, could
            //! build both in so layers and connections is hardcoded (no).
            //!
            //! connection_spec looks like:
            //! ([0], 1) layer 0 connects to layer 1
            //! ([0], 2) layer 0 connects to layer 2
            //! ([1,2], 3) layers 1 & 2 connect to layer 3
            //! ([1,2], 4) layers 1 & 2 connect to layer 4
            //! ([1,2], 5) layers 1 & 2 connect to layer 5
            //! ([5,4], 1) layers 4 & 5 connect to layer 1
            //!
            SpecialNet (const std::vector<unsigned int>& layer_spec,
                        const std::vector<std::pair<morph::vVector<unsigned int>, unsigned int>>& connection_spec, T _tau)
            {
                // Set up layers according to layer_spec
                for (auto nn : layer_spec) {
                    // Create, and zero, a layer containing nn neurons:
                    morph::vVector<T> lyr(nn);
                    lyr.zero();
                    this->p_inputs.push_back (lyr); // input from a connection network
                    this->p_activations.push_back (lyr); // internal neuron activation - forms a memory
                    this->p_outputs.push_back (lyr); // The transferred output of the population - fed to connection networks
                }
                this->tau.resize(layer_spec.size());
                this->tau.set_from (_tau);
                //std::cout << "tau vector: " << this->tau << " set from " << _tau <<  std::endl;
                this->noise_gain.resize(layer_spec.size());
                this->noise_gain.set_from (T{0});

                // Add the connections according to connection_spec
                for (auto conn : connection_spec) {
                    // conn.first is a vector of input neuron layer indices
                    // conn.second is the single output neuron layer index
                    for (auto iconn : conn.first) {
                        auto l1 = this->p_outputs.begin();
                        for (size_t i = 0; i < iconn; ++i, ++l1) {}
                        auto l2 = this->p_inputs.begin();
                        for (size_t i = 0; i < conn.second; ++i, ++l2) {}
                        std::cout << "Connect " << iconn << "(pop. output) to " << conn.second << " (pop. input)"<< std::endl;
                        morph::nn::NetConn<T> c(&*l1, &*l2);

                        // Set weights up
                        c.setweight (T{0.5});
                        c.setbias (T{0});

                        this->connections.push_back (c);
                    }
                }
            }
            //! Output the network as a string
            std::string str() const
            {
                std::stringstream ss;
                unsigned int i = 0;
                auto c = this->connections.begin();
                for (auto n : this->p_activations) {
                    if (i>0 && c != this->connections.end()) {
                        ss << *c++;
                    }
                    ss << "Layer " << i++ << ":  "  << n << "\n";
                }
                ss << "Target output: " << this->desiredOutput << "\n";
                ss << "Delta out: " << this->delta_out << "\n";
                ss << "Cost:      " << this->cost << "\n";
                return ss.str();
            }

            //! Update the network's outputs from its inputs
            void feedforward()
            {
                // Copy result from each connection to inputs of populations first.
                for (auto& c : this->connections) {
                    std::vector<T>& _z = c.z; // FIXME. IF _z is morph::vVector, can't
                                              // call _out.set_from(_z). That's a fix
                                              // for morph/vVector.h
                    morph::vVector<T>& _out = *(c.out);
                    _out.set_from (_z);
                }

                // For a neural net with neuron models, the first stop is to go through
                // this->neurons calling update() on each layer.
                auto p_in = this->p_inputs.begin();
                auto p_act = this->p_activations.begin();
                auto p_out = this->p_outputs.begin();
                // Opportunity for parallel ops here, but prob. not worthwhile
                //for (auto& p_act : this->p_activations) {
                for (size_t i = 0; i < this->tau.size(); ++i) {
                    // Apply inputs to act (da/dt
                    (*p_act) += (*p_in - *p_act)/this->tau[i];
                    // Apply transfer function to set the output
                    for (size_t i = 0; i < p_out->size(); ++i) {
                        (*p_out)[i] = (*p_act)[i] > T{0} ? std::tanh((*p_act)[i]) : T{0};
                        std::cout << "*p_out[i="<<i<<"] = " << (*p_out)[i] << " from activation = " << (*p_act)[i] << std::endl;
                    }
                }

                // Then run through the connections.
                for (auto& c : this->connections) {
                    std::cout << "Calling c.feedforward()...\n";
                    c.feedforward();
                    // Will copy the output of each connection onto the input of each population in the next feedforward step
                }
            }

            //! A function which shows the difference between the network output and
            //! desiredOutput for debugging
            void evaluate (const std::vector<morph::vVector<float>>& ins,
                           const std::vector<morph::vVector<float>>& outs)
            {
                auto op = outs.begin();
                for (auto ir : ins) {
                    // Set input and output
                    this->p_outputs.front() = ir;
                    this->desiredOutput = *op++;
                    // Compute network and cost
                    this->feedforward();
                    float c = this->computeCost();
                    std::cout << "Input " << ir << " --> " << this->p_outputs.back() << " cf. " << this->desiredOutput << " (cost: " << c << ")\n";
                }
            }

            //! Evaluate against the Mnist test image set
            unsigned int evaluate (const std::multimap<unsigned char, morph::vVector<float>>& testData, int num=10000)
            {
                // For each image in testData, compute cost
                float evalcost = 0.0f;
                unsigned int numMatches = 0;
                int count = 0;
                for (auto img : testData) {
                    unsigned int key = static_cast<unsigned int>(img.first);
                    // Set input
                    this->p_outputs.front() = img.second;
                    // Set output
                    this->desiredOutput.zero();
                    this->desiredOutput[key] = 1.0f;
                    // Update
                    this->feedforward();
                    evalcost += this->computeCost();
                    // Success?
                    if (this->p_outputs.back().argmax() == key) {
                        ++numMatches;
                    }
                    ++count;
                    if (count >= num) {
                        break;
                    }
                }
                return numMatches;
            }
#if 0
            //! Determine the error gradients by the backpropagation method. NB: Call
            //! computeCost() first
            void backprop()
            {
                // Notation follows http://neuralnetworksanddeeplearning.com/chap2.html
                // The output layer is special, as the error in the output layer is given by
                //
                // delta^L = grad_a(C) 0 sigma_prime(z^L)
                //
                // whereas for the intermediate layers
                //
                // delta^l = w^l+1 . delta^l+1 0 sigma_prime (z^l)
                //
                // (where 0 signifies hadamard product, as implemented by vVector's operator*)
                // delta = dC_x/da() * sigmoid_prime(z_out)
                auto citer = this->connections.end();
                --citer; // Now points at output layer
                citer->backprop (this->delta_out); // Layer L delta computed
                // After the output layer, loop through the rest of the layers:
                for (;citer != this->connections.begin();) {
                    auto citer_closertooutput = citer--;
                    // Now citer is closer to input
                    citer->backprop (citer_closertooutput->deltas[0]);
                }
            }
#endif
            //! Set up a population's current activations along with desired output
            void setInput (const morph::vVector<T>& theInput, const morph::vVector<T>& theOutput)
            {
                *(this->p_activations.begin()) = theInput;
                this->desiredOutput = theOutput;
            }

#if 0
            //! Compute the cost for one input and one desired output
            T computeCost()
            {
                // Here is where we compute delta_out:
                this->delta_out = (this->p_outputs.back()-desiredOutput) * (this->connections.back().sigmoid_prime_z_lplus1());
                // And the cost:
                T l = (desiredOutput-this->neurons.back()).length();
                this->cost = T{0.5} * l * l;
                return this->cost;
            }
#endif
            //! What's the cost function of the current output? Computed in computeCost()
            T cost = T{0};

            //! A variable number of neuron layers, each of variable size. In this very
            //! simple network, each 'neuron' is just an activation of type T. Note that
            //! the connectivity between the layers is NOT assumed to be simple feedforward.
            std::list<morph::vVector<T>> p_inputs; // input to a population
            std::list<morph::vVector<T>> p_activations; // population activations
            std::list<morph::vVector<T>> p_outputs; // population outputs

            // A series of vectors with parameters for now
            morph::vVector<T> tau; // Time constant for each neuron population
            morph::vVector<T> noisegain; // noise gain for each neuron population

            //! Connections in the network.
            std::list<morph::nn::NetConn<T>> connections;

            //! The error (dC/dz) of the output layer
            morph::vVector<T> delta_out;
            //! The desired output of the network
            morph::vVector<T> desiredOutput;
        };

        template <typename T>
        std::ostream& operator<< (std::ostream& os, const morph::nn::SpecialNet<T>& ff)
        {
            os << ff.str();
            return os;
        }
    } // namespace nn
} // namespace morph

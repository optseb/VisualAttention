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
                this->noisegain.resize(layer_spec.size());
                this->noisegain.set_from (T{0});

                // Add the connections according to connection_spec
                for (auto conn : connection_spec) {
                    // conn.first is a vector of input neuron layer indices
                    // conn.second is the single output neuron layer index
                    for (auto iconn : conn.first) {
                        auto l1 = this->p_outputs.begin();
                        for (size_t i = 0; i < iconn; ++i, ++l1) {}
                        auto l2 = this->p_inputs.begin();
                        for (size_t i = 0; i < conn.second; ++i, ++l2) {}
                        morph::nn::NetConn<T> c(&*l1, &*l2);
                        // Set weights up
                        c.setweight_onetoone (T{0.01});
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
                    ss << "Layer activation " << i++ << ":  "  << n << "\n";
                }
                i = 0;
                for (auto n : this->p_outputs) {
                    ss << "Layer output " << i++ << ":  "  << n << "\n";
                }
                return ss.str();
            }

            //! Update the network's outputs from its inputs
            void feedforward()
            {
                // Copy result from each connection to inputs of populations first.
                for (auto& c : this->connections) {
                    std::vector<T>& _z = c.z;
                    c.out->set_from (_z);
                }

                // Update neurons on each layer
                auto p_in = this->p_inputs.begin();
                auto p_act = this->p_activations.begin();
                auto p_out = this->p_outputs.begin();
                // Opportunity for parallel ops here, but prob. not worthwhile
                for (size_t i = 0; i < this->tau.size(); ++i) {
                    // Apply inputs to act (da/dt)
                    (*p_act) += (*p_in - *p_act)/this->tau[i];
                    // Apply transfer function to set the output
                    for (size_t i = 0; i < p_out->size(); ++i) {
                        (*p_out)[i] = (*p_act)[i] > T{0} ? std::tanh((*p_act)[i]) : T{0};
                    }
                    ++p_in; ++p_out; ++p_act;
                }

                // Then run through the connections.
                for (auto& c : this->connections) { c.feedforward(); }
            }

            //! Set up a population's current activations along with desired output
            void setInput (const morph::vVector<T>& theInput, const morph::vVector<T>& theOutput)
            {
                *(this->p_activations.begin()) = theInput;
                auto p_act = this->p_activations.begin();
                this->p_outputs.resize (this->p_activations.size());
                auto p_out = this->p_outputs.begin();
                // Apply transfer function to set the output based on these activations
                for (size_t i = 0; i < p_out->size(); ++i) {
                    (*p_out)[i] = (*p_act)[i] > T{0} ? std::tanh((*p_act)[i]) : T{0};
                }
                this->desiredOutput = theOutput;
            }

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

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
#include <string>

namespace morph {
    namespace nn {

        template <typename T>
        struct population
        {
            population(unsigned int sz)
            {
                this->inputs.resize(sz);
                this->inputs.zero();
                this->outputs.resize(sz);
                this->outputs.zero();
            }
            // inputs. A vVector of inputs, one for each neural element
            morph::vVector<T> inputs;
            // neuron outputs
            morph::vVector<T> outputs;
            // The population interface - an update() function
            virtual void update() = 0;
            // For output, show just the population's current outputs
            std::string str() const
            {
                std::stringstream ss;
                ss << "Pop outputs: " << outputs << std::endl;
                return ss.str();
            }
        };
        template <typename T>
        std::ostream& operator<< (std::ostream& os, const morph::nn::population<T>& p)
        {
            os << p.str();
            return os;
        }

        template <typename T>
        struct lin_population : public population<T>
        {
            lin_population(unsigned int sz, T _tau)
                : population<T>::population (sz)
                , tau(_tau)
            {
                this->acts.resize (sz);
                this->acts.zero();
            }
            // Leaky integrators need neuron activations
            morph::vVector<T> acts;
            // Time constant
            T tau = T{10};
            // A noise gain
            T noisegain = T{0};
            // Leaky integrator-specific update function
            virtual void update()
            {
                // Apply inputs to act (da/dt)
                this->acts += (this->inputs - this->acts)/this->tau;
                // Apply transfer function to set the output
                for (size_t i = 0; i < this->outputs.size(); ++i) {
                    this->outputs[i] = this->acts[i] > T{0} ? std::tanh(this->acts[i]) : T{0};
                }
            }
        };

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
                        const std::vector<std::pair<morph::vVector<unsigned int>,
                        unsigned int>>& connection_spec, T _tau)
            {
                // Set up layers according to layer_spec
                for (auto nn : layer_spec) {
                    // Create a population containing nn neurons:
                    morph::nn::lin_population<T> p(nn, _tau);
                    this->pops.push_back (p);
                }

                // Add the connections according to connection_spec
                for (auto conn : connection_spec) {
                    // conn.first is a vector of input neuron layer indices
                    // conn.second is the single output neuron layer index
                    for (auto iconn : conn.first) {
                        // Find output and input populations
                        auto lo = this->pops.begin();
                        for (size_t i = 0; i < iconn; ++i, ++lo) {}
                        auto li = this->pops.begin();
                        for (size_t i = 0; i < conn.second; ++i, ++li) {}
                        morph::nn::NetConn<T> c(&(lo->outputs), &(li->inputs));
                        // Set weights up as onetoone. Can set up later too.
                        c.setweight_onetoone (T{0.01});
                        c.setbias (T{0});
                        this->connections.push_back (c);
                    }
                }
            }

            //! Update the network's outputs from its inputs
            void feedforward()
            {
                // Zero each population's inputs
                for (auto& p : this->pops) { p.inputs.zero(); }
                // Now copy previous step's result from each connection to inputs of populations
                for (auto& c : this->connections) { *(c.out) += c.z;  }
                // Update the population neuron models
                for (auto& p : this->pops) { p.update(); }
                // Then run through the connections, doing weights * inputs for each
                for (auto& c : this->connections) { c.feedforward(); }
            }

            //! Set up the first population's current activations
            void setinput (const morph::vVector<T>& theInput)
            {
                this->pops.begin()->inputs = theInput;
                this->pops.begin()->acts = theInput;
                this->pops.begin()->update();
            }

            //! Output the network as a string
            std::string str() const
            {
                std::stringstream ss;
                unsigned int i = 0;
                for (auto c : this->connections) {
                    ss << "Connection " << i++ << ": " << c << "\n";
                }
                i = 0;
                for (auto p : this->pops) {
                    ss << "Population " << i++ << ":  "  << p << "\n";
                }
                return ss.str();
            }

            //! Leaky integrator populations
            std::list<lin_population<T>> pops;
            //! Connections in the network.
            std::list<morph::nn::NetConn<T>> connections;
        };

        template <typename T>
        std::ostream& operator<< (std::ostream& os, const morph::nn::SpecialNet<T>& ff)
        {
            os << ff.str();
            return os;
        }
    } // namespace nn
} // namespace morph

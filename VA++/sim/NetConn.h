/*
 * This file contains a class to hold the information about the connections between
 * layers of neurons in the network.
 *
 * Developed from FeedForwardConn.h with intention of generalising to non-perceptron models.
 *
 * \author Seb James
 * \date March 2022
 */
#pragma once

#include <morph/vVector.h>
#include <iostream>
#include <sstream>
#include <ostream>
#include <vector>

namespace morph {
    namespace nn {

        /*
         * A connection between neuron layers in a feed forward neural network. This
         * connects any number of input neuron populations to a single output
         * population.
         */
        template <typename T>
        struct NetConn
        {
            // Construct for connection from single input layer to single output layer
            NetConn (morph::vVector<T>* _in, morph::vVector<T>* _out)
            {
                this->ins.resize(1);
                this->ins[0] = _in;
                this->commonInit (_out);
            }

            // Construct for connection from many input layers to single output layer
            NetConn (std::vector<morph::vVector<T>*> _ins, morph::vVector<T>* _out)
            {
                this->ins = _ins;
                this->commonInit (_out);
            }

            // Init common to all constructors
            void commonInit (morph::vVector<T>* _out)
            {
                this->out = _out;
                this->N = _out->size();

                this->deltas.resize (this->ins.size());
                for (unsigned int i = 0; i < this->deltas.size(); ++i) {
                    this->deltas[i].resize (this->ins[i]->size(), T{0});
                }

                this->ws.resize (this->ins.size());
                for (unsigned int i = 0; i < this->ws.size(); ++i) {
                    this->ws[i].resize(this->ins[i]->size() * this->N, T{0});
                }

                this->nabla_ws.resize (this->ins.size());
                for (unsigned int i = 0; i < this->nabla_ws.size(); ++i) {
                    this->nabla_ws[i].resize(this->ins[i]->size() * this->N, T{0});
                    this->nabla_ws[i].zero();
                }

                this->b.resize (N, T{0});
                this->nabla_b.resize (N, T{0});
                this->nabla_b.zero();
                this->z.resize (N, T{0});
                this->z.zero();
            }

            // Input layer has total size M = m1 + m2 +... etc where m1, m2 are the lengths of the elements of ins
            std::vector<morph::vVector<T>*> ins; // Each input is the output of a SpecialNet population

            // Old desc: Activation of the output neurons. Computed in feedforward, used in backprop
            // z = sum(w.in) + b. Final output written into *out is the sigmoid(z). Size N.
            // So in this scheme activations are HERE and these are the activations of the output layer...
            //
            // The 'activation' of this connection. This is the weights-times-the-inputs
            // and becomes the output of this connection net that is fed into the output
            // population's input. Glad we got that straight.
            morph::vVector<T> z;

            // This points to the output population's input storage.
            morph::vVector<T>* out;

            // The size (i.e. number of neurons) in z.
            size_t N = 0;
            // The errors in the input layer of neurons. Size M = m1 + m2 +...
            std::vector<morph::vVector<T>> deltas;
            // Weights.
            // Order of weights: w_11, w_12,.., w_1M, w_21, w_22, w_2M, etc. Size M by N = m1xN + m2xN +...
            std::vector<morph::vVector<T>> ws;
            // Biases. Size N. Used in feedforward nets. Here they could be used for connection noise.
            morph::vVector<T> b;
            // The gradients of cost vs. weights. Size M by N = m1xN + m2xN +...
            std::vector<morph::vVector<T>> nabla_ws;
            // The gradients of cost vs. biases. Size N.
            morph::vVector<T> nabla_b;

            // Output as a string
            std::string str() const
            {
                std::stringstream ss;
                ss << "Connection: From " << ins.size() << " input layers of sizes ";
                for (auto _in : ins) {
                    ss << _in->size() << ", ";
                }
                ss << "to an output for this connection of size " << z.size() << "\n";
                size_t ci = 0;
                for (auto w : this->ws) {
                    ss << " Input " << ci++ << ": Weights: w" << w << "w (" << w.size() << ")\n";
                }
                ss << "z = " << z << std::endl;
#if 0
                ci = 0;
                for (auto nabla_w : this->nabla_ws) {
                    ss << " Input " << ci++ << ": nabla_w:nw" << nabla_w << "nw (" << nabla_w.size() << ")\n";
                }
                ss << " Output Biases: b" << b << "b (" << b.size() << ")\n";
                ss << " Output nabla_b:nb" << nabla_b << "nb (" << nabla_b.size() << ")\n";
                ci = 0;
                for (auto delta : this->deltas) {
                    ss << " Input " << ci++ << ": delta  :  " << delta << "\n";
                }
#endif
                return ss.str();
            }

            // Randomize the weights and biases
            void randomize()
            {
                for (auto& w : this->ws) { w.randomizeN (T{0.0}, T{1.0}); }
                this->b.randomizeN (T{0.0}, T{1.0});
            }

            // Set weight/bias to a scalar for all connections
            void setweight_alltoall (T _w) { for (auto& w : this->ws) { w.set_from(_w); } }
            // Ok, so I really need the weight setting code even to debug this first version...
            void setweight_onetoone (T _w) {
                for (auto& w : this->ws) { // w is a morph::vVector<T>
                    for (size_t i = 0; i < (w.size()/this->N); i++) {
                        w[(i*this->N)+i] = _w;
                    }
                }
            }

            void setbias(T _b) { this->b.set_from(_b); }

            // Feed-forward compute. z[i] = in[0,..,M-1] . w[i,..,i+M-1] + b[i] (but
            // have to loop over each input population)
            void feedforward()
            {
                std::cout << "NetConn::feedforward()\n";
                // First, set the output of this connection, z to 0
                this->z.zero();

                // Loop over input populations:
                for (size_t i = 0; i < this->ins.size(); ++i) {
                    // A morph::vVector for a 'part of w'
                    std::cout << "*this->ins["<<i<<"] = " << *this->ins[i] << std::endl;
                    morph::vVector<T>* _in = this->ins[i];
                    size_t m = _in->size();// Size m[i]
                    std::cout << "m = " << m << std::endl;
                    morph::vVector<T> wpart(m);
                    // Get weights, outputs and biases iterators
                    auto witer = this->ws[i].begin();
                    // Carry out an N sized for loop computing each output
                    for (size_t j = 0; j < this->N; ++j) { // Each output
                        // Copy part of weights to wpart (M elements):
                        std::copy (witer, witer+m, wpart.begin());
                        std::cout << "wpart: " << wpart << " dot " << (*ins[i]) << std::endl;
                        // Compute/accumulate dot product with input
                        this->z[j] += wpart.dot (*ins[i]);
                        std::cout << "Set z["<<j<<"] to " << z[j] << std::endl;
                        // Move to the next part of the weight matrix for the next loop
                        witer += m;
                    }
                }

                // For each activation, z, apply the transfer function to generate the output, out
                this->applyTransfer(); // Now this ONLY adds the biases
            }

            // For each activation, z, add the bias, then apply the sigmoid transfer function
            void applyTransfer()
            {
                //auto oiter = this->out->begin();
                auto biter = this->b.begin();
                for (size_t j = 0; j < this->N; ++j) {
                    this->z[j] += *biter++;
                    //*oiter++ = T{1} / (T{1} + std::exp(-z[j])); // out = sigmoid(z+bias)
                }
            }

            // The content of *NetConn::out is sigmoid(z^l+1). \return has size N
            // morph::vVector<T> sigmoid_prime_z_lplus1() { return (*out) * (-(*out)+T{1}); }

            // The content of *NetConn::in is sigmoid(z^l). \return has size M = m1 + m2 +...
            std::vector<morph::vVector<T>> sigmoid_prime_z_l()
            {
                std::vector<morph::vVector<T>> rtn (this->ins.size());
                for (size_t i = 0; i < this->ins.size(); ++i) {
                    rtn[i] = (*ins[i]) * (-(*ins[i])+T{1});
                }
                return rtn;
            }

#if 0 // No backprop for now with this network
            /*
             * Before calling backprop, work out which of the inputs in the 'next'
             * connection layer is relevant to the output of this connection layer.
             */
            void backprop (const NetConn& conn_nxt)
            {
                // For each input in conn_nxt, compare with our output. That will give
                // us the index to use.
                size_t idx = 0;
                size_t idx_max = conn_nxt.ins.size();
                for (size_t i = 0; i < idx_max; ++i) {
                    if (conn_nxt.ins[i] == this->out) {
                        idx = i;
                        break;
                    }
                }
                this->backprop (conn_nxt.deltas[idx]);
            }

            /*
             * Compute this->delta using the values computed in NetConn::feedforward
             * (which must have been executed beforehand).
             */
            void backprop (const morph::vVector<T>& delta_l_nxt) // delta_l_nxt has size N.
            {
                // Check sum of sizes in delta_l_nxt
                if (delta_l_nxt.size() != this->out->size()) {
                    std::stringstream ee;
                    ee << "backprop: Mismatched size. delta_l_nxt size: "
                       << delta_l_nxt.size() << ", out size: " << this->out->size();
                    throw std::runtime_error (ee.str());
                }

                // we have to do weights * delta_l_nxt to give a morph::vVector<T>
                // result. This is the equivalent of the matrix multiplication:
                std::vector<morph::vVector<T>> w_times_deltas(this->ins.size());
                for (size_t idx = 0; idx < this->ins.size(); ++idx) {
                    size_t m = this->ins[idx]->size();
                    w_times_deltas[idx].resize(m);
                    w_times_deltas[idx].zero();
                    for (size_t i = 0; i < m; ++i) { // Each input
                        for (size_t j = 0; j < this->N; ++j) { // Each output
                            // For each weight fanning into neuron j in l_nxt, sum up:
                            w_times_deltas[idx][i] += this->ws[idx][i+(m*j)] * delta_l_nxt[j];
                        }
                    }
                }

                 // spzl has size M; deriv of input
                std::vector<morph::vVector<T>> spzl = this->sigmoid_prime_z_l();

                if (spzl.size() < this->deltas.size()) {
                    throw std::runtime_error ("Sizes error (spzl and deltas)");
                }

                for (size_t idx = 0; idx < this->deltas.size(); ++idx) {
                    this->deltas[idx] = w_times_deltas[idx] * spzl[idx];
                }

                // NB: In a given connection, we compute nabla_b and nabla_w relating to the
                // *output* neurons and the weights also related to the output neurons.

                this->nabla_b = delta_l_nxt; // Size is N

                for (size_t idx = 0; idx < this->ins.size(); ++idx) {
                    size_t m = this->ins[idx]->size();
                    for (size_t i = 0; i < m; ++i) { // Each input
                        for (size_t j = 0; j < this->N; ++j) { // Each output
                            // nabla_w is a_in * delta_out:
                            this->nabla_ws[idx][i+(m*j)] = (*(ins[idx]))[i] * delta_l_nxt[j];
                        }
                    }
                }
            }
#endif
        };

        // Stream operator
        template <typename T>
        std::ostream& operator<< (std::ostream& os, const NetConn<T>& c)
        {
            os << c.str();
            return os;
        }

    } // namespace nn
} // namespace morph

#pragma once

namespace morph {
    namespace nn {
        // One, individual connection from index i to index j. A single line in a weight table
        template <typename T>
        struct conn
        {
            unsigned int i = 0;
            unsigned int j = 0;
            // Omit a per-connection delay
            T w = T{1}; // The weight from input neuron i to output neuron j
        };
    }
}

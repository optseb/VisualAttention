#pragma once

#include <iostream>
#include <vector>
#include <cmath>
#include <sstream>
#include <morph/tools.h>
#include <morph/Vector.h>
#include <morph/HexGrid.h>
#include "WeightHdfData.h"
#include "conn.h"

// This is a bit like a SpineCreator connectionFunc. Here, we connect from one HexGrid
// to another with a Gaussian projection defined by the parameters in sigma. Result
// returned in weight_table. Offset allows you to shift the Gaussian's field around
std::vector<morph::nn::conn<float>> create_gaussian (const morph::HexGrid& p1,
                                                     const morph::HexGrid& p2,
                                                     const morph::Vector<float, 2> sigma,
                                                     const morph::Vector<float, 2> offset = {0, 0},
                                                     const float wco = 0.001f)
{
    std::stringstream dss;
    dss << "gauss_hg" << p1.num() << "_hg" << p2.num() << "_" << sigma[0] <<  "_" << sigma[1] << "_" << wco << ".h5";

    std::vector<morph::nn::conn<float>> weight_table;
    if (morph::Tools::fileExists (dss.str())) {
        WeightHdfData d (dss.str(), morph::FileAccess::ReadOnly);
        std::cout << "Load gauss " << dss.str() << std::endl;
        d.read_weighttable ("/wt", weight_table);
    } else {
        morph::Vector<float, 2> threesig = 3.0f * sigma;
        morph::Vector<float, 2> params = 1.0f / (2.0f * sigma * sigma);
        // Connection is from p1 to p2. Loop through the output first.
        for (auto h2 : p2.hexen) { // outputs
            for (auto h1 : p1.hexen) { // inputs
                // h2.x - h1.x and h
                float d_x = h2.x - h1.x + offset[0];
                float d_y = h2.y - h1.y + offset[1];
                if (d_x < threesig[0] && d_y < threesig[1]) {
                    float w = std::exp ( - ( (params[0] * d_x * d_x) + (params[1] * d_y * d_y) ) );
                    if (w >= wco) {
                        morph::nn::conn<float> c = {h1.vi, h2.vi, w};
                        weight_table.push_back (c);
                    }
                }
            }
        }
        WeightHdfData d (dss.str(), morph::FileAccess::TruncateWrite);
        std::cout << "Save gauss " << dss.str() << std::endl;
        d.add_weighttable ("/wt", weight_table);
    }
    std::cout << "Gaussian weight table size: " << weight_table.size()
              << " (" << (weight_table.size() / p1.num()) << " connections per neuron)\n";
    return weight_table;
}

// A Gabor projection
//
// sigma_g - sigma of the gaussian
// gain_g - a gain for the gaussian
// lambda_s - wavelength of sine
// gain_s - gain of sine
// dir_s - direction of (1-D) sine in degrees
std::vector<morph::nn::conn<float>> create_gabor (const morph::HexGrid& p1,
                                                  const morph::HexGrid& p2,
                                                  const float sigma_g,
                                                  const float gain_g,
                                                  const float lambda_s,
                                                  const float gain_s,
                                                  const float dir_s,
                                                  const float wco = 0.001f)
{
    // Create a filename
    std::stringstream dss;
    dss << "gabor_hg" << p1.num() << "_hg" << p2.num() << "_" << sigma_g << "_" << gain_g << "_" << lambda_s << "_" << gain_s << "_" << dir_s << "_" << wco << ".h5";

    // Return object
    std::vector<morph::nn::conn<float>> weight_table;

    if (morph::Tools::fileExists (dss.str())) {
        WeightHdfData d (dss.str(), morph::FileAccess::ReadOnly);
        std::cout << "Load gabor " << dss.str() << std::endl;
        d.read_weighttable ("/wt", weight_table);
    } else {
        // Connection is from p1 to p2. Loop through the output first.
        for (auto h2 : p2.hexen) { // outputs
            for (auto h1 : p1.hexen) { // inputs
                // h2.x - h1.x and h
                float d_x = h2.x - h1.x;
                float d_y = h2.y - h1.y;
                float dist = std::sqrt (d_x*d_x + d_y*d_y);
                // Direction from source to dest
                float dir_d = std::atan2 (d_y, d_x);
                // Find the projection of the source->dest direction onto the sine wave direction. Call this distance dprime.
                float dprime = dist * std::cos (dir_d + morph::mathconst<float>::two_pi
                                                - ((dir_s * morph::mathconst<float>::two_pi)/360.0f));
                // Use dprime to figure out what the sine weight is.
                float sine_weight = gain_s * std::sin (dprime * morph::mathconst<float>::two_pi / lambda_s);
                float gauss_weight = gain_g * std::exp(-0.5f * (dist/sigma_g) * (dist/sigma_g));
                float combined_weight = sine_weight * gauss_weight;
                if (std::abs(combined_weight) > wco) {
                    morph::nn::conn<float> c = {h1.vi, h2.vi, combined_weight};
                    weight_table.push_back (c);
                }
            }
        }
        WeightHdfData d (dss.str(), morph::FileAccess::TruncateWrite);
        std::cout << "Save gabor " << dss.str() << std::endl;
        d.add_weighttable ("/wt", weight_table);
    }
    std::cout << "Gabor weight table size: " << weight_table.size()
              << " (" << (weight_table.size() / p1.num()) << " connections per neuron)\n";
    return weight_table;
}

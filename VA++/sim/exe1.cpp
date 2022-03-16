// Helloworld for VA++

#include <iostream>
#include <vector>
#include <cmath>

#include <morph/Scale.h>
#include <morph/Vector.h>
#include <morph/Visual.h>
#include <morph/VisualDataModel.h>
#include <morph/HexGridVisual.h>
#include <morph/HexGrid.h>

#include "SpecialNet.h"
#include "conn.h"

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>

// This is a bit like a SpineCreator connectionFunc. Here, we connect from one HexGrid
// to another with a Gaussian projection defined by the parameters in sigma. Result
// returned in weight_table.
std::vector<morph::nn::conn<float>> create_gaussian (const morph::HexGrid& p1,
                                                     const morph::HexGrid& p2,
                                                     const morph::Vector<float, 2> sigma)
{
    std::vector<morph::nn::conn<float>> weight_table;
    morph::Vector<float, 2> threesig = 3.0f * sigma;
    morph::Vector<float, 2> params = 1.0f / (2.0f * sigma * sigma);
    // Connection is from p1 to p2. Loop through the output first.
    for (auto h2 : p2.hexen) { // outputs
        for (auto h1 : p1.hexen) { // inputs
            // h2.x - h1.x and h
            float d_x = h2.x - h1.x;
            float d_y = h2.y - h1.y;
            if (d_x < threesig[0] && d_y < threesig[1]) {
                float w = std::exp ( - ( (params[0] * d_x * d_x) + (params[1] * d_y * d_y) ) );
                morph::nn::conn<float> c = {h1.vi, h2.vi, w};
                weight_table.push_back (c);
            }
        }
    }
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
    std::vector<morph::nn::conn<float>> weight_table;
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
    std::cout << "Gabor weight table size: " << weight_table.size() << " (" << (weight_table.size() / p1.num()) << " connections per neuron)\n";
    return weight_table;
}

int main()
{
    morph::Visual v(1600, 1000, "HexGrid with a neural net", {-0.8,-0.8}, {.05,.05,.05}, 2.0f, 0.0f);

    // at 0.8f, HexGrid with d=0.01f has about same number of elements as the 150x150 grids in the orig. model.
    float gridsize = 0.5f;

    morph::HexGrid hg0(0.01f, 3.0f, 0.0f, morph::HexDomainShape::Boundary);
    hg0.setCircularBoundary (gridsize);
    morph::Vector<float, 3> hg0_loc = { 0.0f, 0.0f, 0.0f };

    morph::HexGrid hg1(0.01f, 3.0f, 0.0f, morph::HexDomainShape::Boundary);
    hg1.setCircularBoundary (gridsize);
    morph::Vector<float, 3> hg1_loc = { -(hg1.width()*0.6f), 0.0f, 0.5f };

    morph::HexGrid hg2(0.01f, 3.0f, 0.0f, morph::HexDomainShape::Boundary);
    hg2.setCircularBoundary (gridsize);
    morph::Vector<float, 3> hg2_loc = { +(hg1.width()*0.6f), 0.0f, 0.5f };
    std::cout << "Number of pixels in grid:" << hg1.num() << std::endl;

    // Create one "Special Network" object, with a connectivity specification.
    std::vector<std::pair<morph::vVector<unsigned int>, unsigned int>> connspec = {{{0},1},{{0},2}};
    float tau = 1000.0f;
    morph::nn::SpecialNet<float> snet({hg0.num(),hg1.num(),hg2.num()}, connspec, tau);

    // Create weight tables and set these into the network's connections
    float gain_g = 1.0f;
    float lambda_s = 0.05f;
    float gain_s = 1.0f;
    float dir_s_1 = 30.0f;
    float dir_s_2 = 120.0f;

    std::vector<morph::nn::conn<float>> weight_table1 = create_gabor (hg0, hg1,
                                                                      hg0.getd()/2.0f,
                                                                      gain_g,
                                                                      lambda_s,
                                                                      gain_s,
                                                                      dir_s_1);
    auto c = snet.connections.begin();
    c->setweight (weight_table1);

    std::vector<morph::nn::conn<float>> weight_table2 = create_gabor (hg0, hg2,
                                                                      hg0.getd()/2.0f,
                                                                      gain_g,
                                                                      lambda_s,
                                                                      gain_s,
                                                                      dir_s_2);
    c++;
    c->setweight (weight_table2);

    // Load an image
    //std::string fn = "../sim/Lbig.png";
    std::string fn = "../sim/bike256.png";
    cv::Mat img = cv::imread (fn.c_str(), cv::IMREAD_GRAYSCALE);
    img.convertTo (img, CV_32F);
    morph::vVector<float> image_data;
    image_data.assign((float*)img.data, (float*)img.data + img.total()*img.channels());
    image_data /= 255.0f;

    morph::Vector<float,2> image_scale = {0.6f, 0.6f}; // what's the scale of the image in HexGrid's units?
    morph::Vector<float,2> image_offset = {0.0f, 0.0f}; // offset in HexGrid's units (if 0, image is centered on HexGrid)
    morph::vVector<float> data0 = hg0.resampleImage (image_data, img.cols, image_scale, image_offset);

    morph::vVector<float> theoutput(hg2.num());
    theoutput.zero();
    auto popout = snet.p_outputs.begin();
    snet.setInput (data0, theoutput);

    snet.feedforward();
    snet.feedforward();

    morph::HexGridVisual<float>* hgv0 = new morph::HexGridVisual<float>(v.shaderprog, v.tshaderprog, &hg0, hg0_loc);
    hgv0->setScalarData (&*popout++);
    hgv0->cm.setType (morph::ColourMapType::GreyscaleInv);
    hgv0->hexVisMode = morph::HexVisMode::HexInterp;
    hgv0->zScale.setParams (0, 1); // This fixes the z scale to have zero slope - so no relief in the map
    hgv0->finalize();
    v.addVisualModel (hgv0);

    morph::HexGridVisual<float>* hgv1 = new morph::HexGridVisual<float>(v.shaderprog, v.tshaderprog, &hg1, hg1_loc);
    hgv1->setScalarData (&*popout++);
    hgv1->hexVisMode = morph::HexVisMode::HexInterp;
    hgv1->zScale.setParams (0, 1);
    hgv1->addLabel ("hgv1", { -0.2f, 0.2f, 0.01f },
                    morph::colour::black, morph::VisualFont::DVSans, 0.1f, 48);
    hgv1->finalize();
    v.addVisualModel (hgv1);

    morph::HexGridVisual<float>* hgv2 = new morph::HexGridVisual<float>(v.shaderprog, v.tshaderprog, &hg2, hg2_loc);
    hgv2->setScalarData (&*popout++);
    hgv2->hexVisMode = morph::HexVisMode::HexInterp; // Or morph::HexVisMode::Triangles for a smoother surface plot
    hgv2->zScale.setParams (0, 1);
    hgv2->addLabel ("hgv2", { -0.2f, 0.2f, 0.01f },
                    morph::colour::black, morph::VisualFont::DVSans, 0.1f, 48);
    hgv2->finalize();
    v.addVisualModel (hgv2);

    size_t loop = 0;
    while (v.readyToFinish == false) {
        snet.feedforward();
        if (loop++%1000 == 0) {
            glfwWaitEventsTimeout (0.001);
            popout = snet.p_outputs.begin();
            hgv0->updateData(&*popout++);
            hgv1->updateData(&*popout++);
            hgv1->clearAutoscaleColour(); // Ensures colour scale re-normalises each time
            hgv2->updateData(&*popout++);
            hgv2->clearAutoscaleColour();
            v.render();
        }
    }
    std::cout << "Computed " << loop << " loops\n";

    return 0;
}

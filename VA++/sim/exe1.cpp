// Helloworld for VA++

#include <iostream>
#include <vector>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <morph/Scale.h>
#include <morph/Vector.h>
#include <morph/Visual.h>
#include <morph/VisualDataModel.h>
#include <morph/HexGridVisual.h>
#include <morph/HexGrid.h>
#include <morph/Config.h>
#include "SpecialNet.h"
#include "conn.h"
#include "projections.h"

int main()
{
    morph::Visual v(1600, 1000, "HexGrids with a neural net", {-0.8,-0.8}, {.05,.05,.05}, 2.0f, 0.0f);

    morph::Config conf("../sim/exe1.json");
    if (!conf.ready) {
        std::cout << "Failed to read config.\n";
        return -1;
    }

    // at 0.8f, HexGrid with d=0.01f has about same number of elements as the 150x150 grids in the orig. model.
    float gridsize = conf.getFloat ("gridsize", 0.5f);
    // Hex to hex size
    float hexhex = conf.getFloat ("hexhex", 0.01f);

    // A single HexGrid is used for the positions of the neurons for all populations of this size.
    morph::HexGrid hg0(hexhex, gridsize * 3.0f, 0.0f, morph::HexDomainShape::Boundary);
    hg0.setCircularBoundary (gridsize);
    unsigned int n0 = hg0.num();
    std::cout << "Number of hexagonal pixels in the HexGrid:" << n0 << std::endl;

    // Create one "Special Network" object, with a connectivity specification. This
    // specifies only the lines between the populations, not what kind of
    // neuron-to-neuron connectivity weight maps there will be.
    std::vector<std::pair<morph::vVector<unsigned int>, unsigned int>> connspec = {
        {{0},1},  {{0},2},  // layer 0 to layer 1
        {{1},2},  {{2},1},  // layer 1 intra
        {{1},3},  {{1},4}, {{1},5}, {{1},6}, {{1},7},  {{2},4}, {{2},5}, {{2},6}, {{2},7},  {{2},8} // l1 to l2
    };
    float tau = conf.getFloat ("tau", 1000.0f);
    morph::nn::SpecialNet<float> snet({n0, n0,n0, n0,n0,n0,n0,n0,n0 }, connspec, tau);

    // Locations for 3 visualisations based on the same grid
    std::vector<morph::Vector<float, 3>> hg_locs;
    // Layer 0
    hg_locs.push_back ({ 0.0f, 0.0f, 0.0f });
    // Layer 1
    hg_locs.push_back ({ -(hg0.width()*0.6f), hg0.width(), 0.0f });
    hg_locs.push_back ({ +(hg0.width()*0.6f), hg0.width(), 0.0f });
    // Layer 2
    hg_locs.push_back ({ -(hg0.width()*1.8f), 2.5f*hg0.width(), 0.0f });
    hg_locs.push_back ({ -(hg0.width()*0.6f), 2.0f*hg0.width(), 0.0f });
    hg_locs.push_back ({ -(hg0.width()*0.6f), 3.0f*hg0.width(), 0.0f });
    hg_locs.push_back ({  (hg0.width()*0.6f), 2.0f*hg0.width(), 0.0f });
    hg_locs.push_back ({  (hg0.width()*0.6f), 3.0f*hg0.width(), 0.0f });
    hg_locs.push_back ({  (hg0.width()*1.8f), 2.5f*hg0.width(), 0.0f });

    // Create weight tables and set these into the network's connections
    float gain_g = conf.getFloat ("gabor_gain_g", 1.0f);
    float lambda_s = conf.getFloat ("gabor_lambda_s", 0.05f);
    float gain_s = conf.getFloat ("gabor_gain_s", 1.0f);
    float dir_s_1 = conf.getFloat ("gabor_dir1", 0.0f);
    float dir_s_2 = conf.getFloat ("gabor_dir2", 90.0f);

    std::vector<morph::nn::conn<float>> weight_table;

    // Note: This is the part of setting up a neural network that really IS better in
    // SpineCreator - the definition of connections between objects.
    auto c = snet.connections.begin();

    // Layer 0 to layer 1
    weight_table = create_gabor (hg0, hg0, hg0.getd()/2.0f, gain_g, lambda_s, gain_s, dir_s_1);
    c->setweight (weight_table);

    weight_table = create_gabor (hg0, hg0, hg0.getd()/2.0f, gain_g, lambda_s, gain_s, dir_s_2);
    c++;
    c->setweight (weight_table);

    // Layer 1 to Layer 1
    float oneone_weight = conf.getFloat ("oneone_weight", -0.6f);
    c++;
    c->setweight_onetoone (oneone_weight);
    c++;
    c->setweight_onetoone (oneone_weight);

    // Layer 1 to layer 2
    float dg = hg0.getd()/2.0f;
    morph::Vector<float, 2> g_sigma = {dg, dg};
    c++;
    weight_table = create_gaussian (hg0, hg0, g_sigma, {dg, 0}); // Will be create_dualgauss()
    c->setweight (weight_table);

    // Choose the offset distance.
    float os = hg0.getd() * conf.getFloat ("offset_gauss_distmult", 6.0f);

    // These four are conjunctions
    c++;
    weight_table = create_gaussian (hg0, hg0, g_sigma, {os, 0});
    c->setweight (weight_table);

    c++;
    weight_table = create_gaussian (hg0, hg0, g_sigma, {os, 0});
    c->setweight (weight_table);

    c++;
    weight_table = create_gaussian (hg0, hg0, g_sigma, {-os, 0});
    c->setweight (weight_table);

    c++;
    weight_table = create_gaussian (hg0, hg0, g_sigma, {-os, 0});
    c->setweight (weight_table);

    // These four are conjunctions
    c++;
    weight_table = create_gaussian (hg0, hg0, g_sigma, {0, os});
    c->setweight (weight_table);

    c++;
    weight_table = create_gaussian (hg0, hg0, g_sigma, {0, -os});
    c->setweight (weight_table);

    c++;
    weight_table = create_gaussian (hg0, hg0, g_sigma, {0, -os});
    c->setweight (weight_table);

    c++;
    weight_table = create_gaussian (hg0, hg0, g_sigma, {0, os});
    c->setweight (weight_table);

    c++;
    //weight_table = create_dualgaussian (hg0, hg0, g_sigma, {0, os});
    c->setweight (weight_table);

    // Load an image
    std::string fn = "../sim/L.png";
    //std::string fn = "../sim/bike256.png";
    cv::Mat img = cv::imread (fn.c_str(), cv::IMREAD_GRAYSCALE);
    img.convertTo (img, CV_32F);
    morph::vVector<float> image_data;
    image_data.assign (reinterpret_cast<float*>(img.data),
                       reinterpret_cast<float*>(img.data) + img.total() * img.channels());
    image_data /= 255.0f;

    float img_sz = conf.getFloat ("image_scale", 0.3f);
    morph::Vector<float,2> image_scale = {img_sz, img_sz}; // what's the scale of the image in HexGrid's units?
    morph::Vector<float,2> image_offset = {0.0f, 0.0f}; // offset in HexGrid's units (if 0, image is centered on HexGrid)
    morph::vVector<float> data0 = hg0.resampleImage (image_data, img.cols, image_scale, image_offset);
    snet.setinput (data0);

    morph::HexGridVisual<float>* hgv = nullptr;
    std::vector<morph::HexGridVisual<float>*> hgvs;

    auto popout = snet.pops.begin();
    for (unsigned int ii = 0; ii < snet.pops.size(); ++ii) {
        std::cout << "Set up hgv for location " << hg_locs[ii] << std::endl;
        hgv = new morph::HexGridVisual<float>(v.shaderprog, v.tshaderprog, &hg0, hg_locs[ii]);
        hgv->setScalarData (&popout++->outputs);
        hgv->cm.setType (morph::ColourMapType::GreyscaleInv);
        hgv->hexVisMode = morph::HexVisMode::HexInterp;
        hgv->zScale.setParams (0, 0);
        hgv->addLabel (std::string("pop")+std::to_string(ii),
                       { -gridsize, gridsize, 0.01f },
                       morph::colour::black, morph::VisualFont::DVSans, 0.1f, 48);
        hgv->finalize();
        v.addVisualModel (hgv);
        hgvs.push_back (hgv);
    }
    std::cout << "size of hgvs: " << hgvs.size() << std::endl;

    // Simulation loop
    size_t loop = 0;
    while (v.readyToFinish == false) {
        snet.feedforward();
        if (loop++%1 == 0) {
            glfwWaitEventsTimeout (0.018);

            popout = snet.pops.begin();
            for (unsigned int ii = 0; ii < snet.pops.size(); ++ii) {
                hgvs[ii]->updateData(&popout++->outputs);
                hgvs[ii]->clearAutoscaleColour(); // Ensures colour scale re-normalises each time
            }

            v.render();
        }
    }
    std::cout << "Computed " << loop << " loops\n";

    return 0;
}

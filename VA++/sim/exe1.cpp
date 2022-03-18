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
#include <morph/HdfData.h>

#include "SpecialNet.h"
#include "conn.h"

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>

// A special HdfData class with a method to save/load std::vector<morph::nn::conn<float>> weight_table
class WeightHdfData : public morph::HdfData
{
public:
    WeightHdfData (const std::string fname,
                   const morph::FileAccess _file_access = morph::FileAccess::TruncateWrite,
                   const bool show_hdf_internal_errors = false)
        : HdfData (fname, _file_access, show_hdf_internal_errors) {}

    void read_weighttable (const char* path, std::vector<morph::nn::conn<float>>& vals)
    {
        // Read vectors, then resize/population the vector<conn<float>>.
        std::string pth(path);
        std::string i_path = pth + "_i";
        std::string j_path = pth + "_j";
        std::string w_path = pth + "_w";
        std::vector<unsigned int> i_array (vals.size());
        std::vector<unsigned int> j_array (vals.size());
        std::vector<float> w_array (vals.size());
        this->read_contained_vals (i_path.c_str(), i_array);
        this->read_contained_vals (j_path.c_str(), j_array);
        this->read_contained_vals (w_path.c_str(), w_array);
        if (i_array.size() != j_array.size() || i_array.size() != w_array.size()) {
            vals.resize(0);
            return;
        }
        vals.resize (i_array.size());
        for (unsigned int ii = 0; ii < i_array.size(); ++ii) {
            vals[ii].i = i_array[ii];
            vals[ii].j = j_array[ii];
            vals[ii].w = w_array[ii];
        }
    }
    void add_weighttable (const char* path, const std::vector<morph::nn::conn<float>>& vals)
    {
        // Each conn contains two unsigned ints and one float value. Make 3 vectors and save those.
        std::vector<unsigned int> i_array (vals.size());
        std::vector<unsigned int> j_array (vals.size());
        std::vector<float> w_array (vals.size());
        for (unsigned int ii = 0; ii < vals.size(); ++ii) {
            i_array[ii] = vals[ii].i;
            j_array[ii] = vals[ii].j;
            w_array[ii] = vals[ii].w;
        }
        std::string pth(path);
        std::string i_path = pth + "_i";
        std::string j_path = pth + "_j";
        std::string w_path = pth + "_w";
        this->add_contained_vals (i_path.c_str(), i_array);
        this->add_contained_vals (j_path.c_str(), j_array);
        this->add_contained_vals (w_path.c_str(), w_array);
    }
};

// This is a bit like a SpineCreator connectionFunc. Here, we connect from one HexGrid
// to another with a Gaussian projection defined by the parameters in sigma. Result
// returned in weight_table.
std::vector<morph::nn::conn<float>> create_gaussian (const morph::HexGrid& p1,
                                                     const morph::HexGrid& p2,
                                                     const morph::Vector<float, 2> sigma)
{
    std::stringstream dss;
    dss << "gauss_hg" << p1.num() << "_hg" << p2.num() << "_" << sigma[0] <<  "_" << sigma[1] << "_" << ".h5";

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
                float d_x = h2.x - h1.x;
                float d_y = h2.y - h1.y;
                if (d_x < threesig[0] && d_y < threesig[1]) {
                    float w = std::exp ( - ( (params[0] * d_x * d_x) + (params[1] * d_y * d_y) ) );
                    morph::nn::conn<float> c = {h1.vi, h2.vi, w};
                    weight_table.push_back (c);
                }
            }
        }
        WeightHdfData d (dss.str(), morph::FileAccess::TruncateWrite);
        std::cout << "Save gauss " << dss.str() << std::endl;
        d.add_weighttable ("/wt", weight_table);
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

int main()
{
    morph::Visual v(1600, 1000, "HexGrid with a neural net", {-0.8,-0.8}, {.05,.05,.05}, 2.0f, 0.0f);

    // at 0.8f, HexGrid with d=0.01f has about same number of elements as the 150x150 grids in the orig. model.
    float gridsize = 0.5f;
    // Hex to hex size
    float hexhex = 0.01f;

    // A single HexGrid is used for the positions of the neurons for all populations of this size.
    morph::HexGrid hg0(hexhex, gridsize * 3.0f, 0.0f, morph::HexDomainShape::Boundary);
    hg0.setCircularBoundary (gridsize);
    std::cout << "Number of hexagonal pixels in the HexGrid:" << hg0.num() << std::endl;

    // Locations for 3 visualisations based on the same grid
    morph::Vector<float, 3> hg0_loc = { 0.0f, 0.0f, 0.0f };
    morph::Vector<float, 3> hg1_loc = { -(hg0.width()*0.6f), 0.0f, 0.5f };
    morph::Vector<float, 3> hg2_loc = { +(hg0.width()*0.6f), 0.0f, 0.5f };


    // Create one "Special Network" object, with a connectivity specification.
    std::vector<std::pair<morph::vVector<unsigned int>, unsigned int>> connspec = {{{0},1},
                                                                                   {{0},2},
                                                                                   {{1},2},
                                                                                   {{2},1}};
    float tau = 1000.0f;
    morph::nn::SpecialNet<float> snet({hg0.num(),hg0.num(),hg0.num()}, connspec, tau);

    // Create weight tables and set these into the network's connections
    float gain_g = 1.0f;
    float lambda_s = 0.05f;
    float gain_s = 1.0f;
    float dir_s_1 = 30.0f;
    float dir_s_2 = 120.0f;

    std::vector<morph::nn::conn<float>> weight_table1 = create_gabor (hg0, hg0,
                                                                      hg0.getd()/2.0f,
                                                                      gain_g,
                                                                      lambda_s,
                                                                      gain_s,
                                                                      dir_s_1);
    auto c = snet.connections.begin();
    c->setweight (weight_table1);

    std::vector<morph::nn::conn<float>> weight_table2 = create_gabor (hg0, hg0,
                                                                      hg0.getd()/2.0f,
                                                                      gain_g,
                                                                      lambda_s,
                                                                      gain_s,
                                                                      dir_s_2);
    c++;
    c->setweight (weight_table2);
    c++;
    c->setweight_onetoone (-0.6);
    c++;
    c->setweight_onetoone (-0.6);

    // Load an image
    //std::string fn = "../sim/Lbig.png";
    std::string fn = "../sim/bike256.png";
    cv::Mat img = cv::imread (fn.c_str(), cv::IMREAD_GRAYSCALE);
    img.convertTo (img, CV_32F);
    morph::vVector<float> image_data;
    image_data.assign (reinterpret_cast<float*>(img.data),
                       reinterpret_cast<float*>(img.data) + img.total() * img.channels());
    image_data /= 255.0f;

    morph::Vector<float,2> image_scale = {0.6f, 0.6f}; // what's the scale of the image in HexGrid's units?
    morph::Vector<float,2> image_offset = {0.0f, 0.0f}; // offset in HexGrid's units (if 0, image is centered on HexGrid)
    morph::vVector<float> data0 = hg0.resampleImage (image_data, img.cols, image_scale, image_offset);

    auto popout = snet.pops.begin();
    snet.setinput (data0);

    snet.feedforward();
//    snet.feedforward();

    morph::HexGridVisual<float>* hgv0 = new morph::HexGridVisual<float>(v.shaderprog, v.tshaderprog, &hg0, hg0_loc);
    hgv0->setScalarData (&popout++->outputs);
    hgv0->cm.setType (morph::ColourMapType::GreyscaleInv);
    hgv0->hexVisMode = morph::HexVisMode::HexInterp;
    hgv0->zScale.setParams (0, 0); // This fixes the z scale to have zero slope - so no relief in the map
    hgv0->finalize();
    v.addVisualModel (hgv0);

    morph::HexGridVisual<float>* hgv1 = new morph::HexGridVisual<float>(v.shaderprog, v.tshaderprog, &hg0, hg1_loc);
    hgv1->setScalarData (&popout++->outputs);
    hgv1->hexVisMode = morph::HexVisMode::HexInterp;
    hgv1->zScale.setParams (0, 0);
    hgv1->addLabel ("hgv1", { -0.2f, 0.2f, 0.01f },
                    morph::colour::black, morph::VisualFont::DVSans, 0.1f, 48);
    hgv1->finalize();
    v.addVisualModel (hgv1);

    morph::HexGridVisual<float>* hgv2 = new morph::HexGridVisual<float>(v.shaderprog, v.tshaderprog, &hg0, hg2_loc);
    hgv2->setScalarData (&popout++->outputs);
    hgv2->hexVisMode = morph::HexVisMode::HexInterp; // Or morph::HexVisMode::Triangles for a smoother surface plot
    hgv2->zScale.setParams (0, 0);
    hgv2->addLabel ("hgv2", { -0.2f, 0.2f, 0.01f },
                    morph::colour::black, morph::VisualFont::DVSans, 0.1f, 48);
    hgv2->finalize();
    v.addVisualModel (hgv2);

    size_t loop = 0;
    while (v.readyToFinish == false) {
        snet.feedforward();
        if (loop++%100 == 0) {
            glfwWaitEventsTimeout (0.018);
            popout = snet.pops.begin();
            hgv0->updateData(&popout++->outputs);
            hgv1->updateData(&popout++->outputs);
            hgv1->clearAutoscaleColour(); // Ensures colour scale re-normalises each time
            hgv2->updateData(&popout++->outputs);
            hgv2->clearAutoscaleColour();
            v.render();
        }
    }
    std::cout << "Computed " << loop << " loops\n";

    return 0;
}

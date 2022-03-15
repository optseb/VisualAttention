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

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>

int main()
{
    morph::Visual v(1600, 1000, "HexGrid with a neural net", {-0.8,-0.8}, {.05,.05,.05}, 2.0f, 0.0f);

    float gridsize = 0.5f; // at 0.8f, neural network uses about 8GB of RAM. At 1.0f 32 GB isn't enough.

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

    // A feedforward network to display with the hexgrids.
    // A feedforward network is probably too simplistic? But feedforward connections are ok.
    std::vector<std::pair<morph::vVector<unsigned int>, unsigned int>> connspec = {{{0},1},{{0},2}};
    float tau = 1000.0f;
    morph::nn::SpecialNet<float> ffn({hg0.num(),hg1.num(),hg2.num()}, connspec, tau);

#if 0
    // data for the input
    morph::vVector<float> data0(hg0.num(), 0.0f);
    for (unsigned int ri=0; ri<hg0.num(); ++ri) {
        data0[ri] = 0.05f + 0.05f*std::sin(20.0f*hg0.d_x[ri]) * std::sin(10.0f*hg0.d_y[ri]) ; // Range 0->1
    }
    //std::cout << "data0 max: " << data0.max() << std::endl;
#else
    // The above to become something like this: where image is some image format
    // (OpenCV?) and w and h set the width and height of the image in the units used on
    // the hex grid (allowing you to scale the size of the resulting resampling). Image
    // is a 1D thing and won't be coloured, so could just be another
    // morph::vVector<float>, though then we'd need to specify pixel width

    // OpenCV code now to create image_data...
    //std::string fn = "../sim/Lbig.png";
    std::string fn = "../sim/bike256.png";
    cv::Mat img = cv::imread (fn.c_str(), cv::IMREAD_GRAYSCALE);
    img.convertTo (img, CV_32F);

    morph::vVector<float> image_data;
    image_data.assign((float*)img.data, (float*)img.data + img.total()*img.channels());

    std::cout << "image_data.size: " << image_data.size()
              << " and min/max: "
              << image_data.min()
              << "/" << image_data.max() << ", hg0 width: " << hg0.width() << std::endl;

    image_data /= 255.0f;

    morph::Vector<float,2> image_scale = {1.1f, 1.1f}; // what's the scale of the image in HexGrid's units?
    morph::Vector<float,2> image_offset = {-0.0f, -0.0f}; // offset in HexGrid's units (if 0, image is centered on HexGrid)
    morph::vVector<float> data0 = hg0.resampleImage (image_data, img.cols, image_scale, image_offset);
#endif

    morph::vVector<float> theoutput(hg2.num());
    theoutput.zero();
    auto popout = ffn.p_outputs.begin();
    ffn.setInput (data0, theoutput);

    //ffn.feedforward();
    //ffn.feedforward();

    morph::HexGridVisual<float>* hgv0 = new morph::HexGridVisual<float>(v.shaderprog, v.tshaderprog, &hg0, hg0_loc);
    hgv0->setScalarData (&*popout++);
    hgv0->cm.setType (morph::ColourMapType::GreyscaleInv);
    hgv0->hexVisMode = morph::HexVisMode::HexInterp;
    hgv0->zScale.setParams (0, 1);
    hgv0->finalize();
    v.addVisualModel (hgv0);

    morph::HexGridVisual<float>* hgv1 = new morph::HexGridVisual<float>(v.shaderprog, v.tshaderprog, &hg1, hg1_loc);
    hgv1->setScalarData (&*popout++);
    hgv1->hexVisMode = morph::HexVisMode::HexInterp;
    hgv1->finalize();
    v.addVisualModel (hgv1);

    morph::HexGridVisual<float>* hgv2 = new morph::HexGridVisual<float>(v.shaderprog, v.tshaderprog, &hg2, hg2_loc);
    hgv2->setScalarData (&*popout++);
    hgv2->hexVisMode = morph::HexVisMode::HexInterp; // Or morph::HexVisMode::Triangles for a smoother surface plot
    hgv2->finalize();
    v.addVisualModel (hgv2);

    //size_t ri = 0;
    while (v.readyToFinish == false) {

        //if (ri < 4) {
#if 0
        ffn.feedforward();
        popout = ffn.p_outputs.begin();
        hgv0->updateData(&*popout++);
        hgv1->updateData(&*popout++);
        hgv2->updateData(&*popout++);
#endif
        //}
        glfwWaitEventsTimeout (0.18); // 0.018 for speed, but slowed down for debug
        v.render();
        //std::cout << "Render " << ri++ << std::endl;
    }

    return 0;
}

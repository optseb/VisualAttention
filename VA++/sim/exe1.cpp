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

int main()
{
    morph::Visual v(1600, 1000, "HexGrid with a neural net", {-0.8,-0.8}, {.05,.05,.05}, 2.0f, 0.0f);

    float gridsize = 0.01f; // at 0.8f, neural network uses about 8GB of RAM. At 1.0f 32 GB isn't enough.

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
    std::vector<std::pair<morph::vVector<unsigned int>, unsigned int>> connspec = {{{0},1}/*,{{0},2}*/};
    float tau = 10.0f;
    morph::nn::SpecialNet<float> ffn({hg0.num(),hg1.num()/*,hg2.num()*/}, connspec, tau);

    // data for the input
    morph::vVector<float> data0(hg0.num(), 0.0f);
    for (unsigned int ri=0; ri<hg0.num(); ++ri) {
        data0[ri] = 0.05f + 0.05f*std::sin(20.0f*hg0.d_x[ri]) * std::sin(10.0f*hg0.d_y[ri]) ; // Range 0->1
    }

    morph::vVector<float> theoutput(hg2.num());
    theoutput.zero();
    auto popout = ffn.p_outputs.begin();
    ffn.setInput (data0, theoutput);

    std::cout << "Before feedforward, ffn: " << ffn << std::endl;
    ffn.feedforward();
    std::cout << "FIRST FEEDFORWARD DONE...\n";
    ffn.feedforward();
    std::cout << "After, ffn: " << ffn << std::endl;

    morph::HexGridVisual<float>* hgv0 = new morph::HexGridVisual<float>(v.shaderprog, v.tshaderprog, &hg0, hg0_loc);
    hgv0->setScalarData (&*popout++);
    hgv0->hexVisMode = morph::HexVisMode::HexInterp;
    hgv0->finalize();
    v.addVisualModel (hgv0);

    morph::HexGridVisual<float>* hgv1 = new morph::HexGridVisual<float>(v.shaderprog, v.tshaderprog, &hg1, hg1_loc);
    hgv1->setScalarData (&*popout/*++*/);
    hgv1->hexVisMode = morph::HexVisMode::HexInterp;
    hgv1->finalize();
    v.addVisualModel (hgv1);

    morph::HexGridVisual<float>* hgv2 = new morph::HexGridVisual<float>(v.shaderprog, v.tshaderprog, &hg2, hg2_loc);
    hgv2->setScalarData (&*popout/*++*/);
    hgv2->hexVisMode = morph::HexVisMode::HexInterp; // Or morph::HexVisMode::Triangles for a smoother surface plot
    hgv2->finalize();
    v.addVisualModel (hgv2);

    while (v.readyToFinish == false) {
        glfwWaitEventsTimeout (0.018);
        v.render();
    }

    return 0;
}

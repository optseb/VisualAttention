# VisualAttention
Visual attention models

Currently a SpineML implementation of the ABRG visual attention model, developed as Alex Cope's PhD with
supervision from Kevin Gurney.

VentralStream/

Contains the SpineML model itself. Note that large files (connection binary files) are revision controlled 
using Git Large File Support, so you will need Git LFS to clone this repository fully.

analysis/

Contains octave/matlab code to visualise the output of the VentralStream model

connectionFuncs/

Python scripts to visualise connection functions which are used in the SpineML model in VentralStream/. It's
easier to visualise outside SpineCreator, then import the connectionFunc() into the SpineCreator environment.

datainput/

C++ code to generate input for the VentralStream model.

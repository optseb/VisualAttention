# Viewing the output of the model in matlab/octave

Add /home/seb/src/SpineCreator/analysis_utils/matlab to your
octave/matlab path. In octave/matlab, run:

```matlab
addpath ('/home/seb/src/SpineCreator/analysis_utils/matlab');
```

(This ensures that octave/matlab can find load_sc_data.m)

Then, to load the data run:

```matlab
A = load_vs ('/home/seb/src/SpineML_2_BRAHMS/temp/Ventral_Stream_e0');
```

Then with the data that is now stored in the struct A, view some
timestep (here, I choose timestep 100):

```matlab
vsview (A, 100)
```

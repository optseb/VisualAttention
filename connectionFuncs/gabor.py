#
# 1) Define the connectionFunc - the code that will be pasted into the
# python script settings window in SpineCreator.
#

#PARNAME=sigma_g  #LOC=1,1
#PARNAME=gain_g   #LOC=1,2
#PARNAME=lambda_s #LOC=2,1
#PARNAME=gain_s   #LOC=2,2
#PARNAME=dir_s    #LOC=3,1
#PARNAME=wco      #LOC=3,2
#PARNAME=roi      #LOC=4,1
#HASWEIGHT

# This implements a Gabor connection; a 2-D Gaussian multiplied by a
# 1-D sine. Arguments are:
#
# sigma_g - sigma of the gaussian
# gain_g - a gain for the gaussian
# lambda_s - wavelength of sine
# gain_s - gain of sine
# dir_s - direction of (1-D) sine in degrees
def connectionFunc(srclocs,dstlocs,sigma_g,gain_g,lambda_s,gain_s,dir_s,wco,roi):

    import math

    twopi = 6.283185307;

    i = 0 # i is 'src iterator'
    j = 0 # j is 'dst iterator'
    out = [] # To return result

    for srcloc in srclocs:
        j = 0
        for dstloc in dstlocs:

            # Avoid many tanh, sin, cos, pow and exp computations for well-separated neurons:
            xdist = srcloc[0] - dstloc[0]
            ydist = srcloc[1] - dstloc[1]
            # Ignore zdist
            if abs(xdist) > roi or abs(ydist) > roi:
                j = j + 1
                continue

            zdist = srcloc[2] - dstloc[2]
            dist = math.sqrt(math.pow(xdist,2) + math.pow(ydist,2) + math.pow(zdist,2))

            # Direction from source to dest
            ##print ('dstloc[1]: ', dstloc[1], ' srcloc[1]:', srcloc[1])
            ##print ('dstloc[0]: ', dstloc[0], ' srcloc[0]:', srcloc[0])
            top = dstloc[1]-srcloc[1]
            bot = dstloc[0]-srcloc[0]
            ##print ('top:',top,'bot:',bot)
            dir_d = math.atan2(top, bot);
            ##print ('dir_d: ', dir_d, 'dir_s:', dir_s)

            # Find the projection of the source->dest direction onto the sine wave direction. Call this distance dprime.
            dprime = dist*math.cos(dir_d + twopi - ((dir_s*twopi)/360));
            ##print ('dprime: ', dprime, ' dist: ', dist)

            # Use dprime to figure out what the sine weight is.
            sine_weight = gain_s*math.sin(dprime*twopi/lambda_s);
            ##print ('sine_weight:', sine_weight)

            gauss_weight = gain_g*math.exp(-0.5*math.pow(dist/sigma_g,2))
            ##print ('gauss_weight:', gauss_weight)

            combined_weight = sine_weight * gauss_weight;
            ##print ('combined_weight:', combined_weight)

            if abs(combined_weight) > wco:
                #sys.stdout.write('gauss>0.0001: i={0} gauss={1}'.format( i, gauss))
                conn = (i,j,0,combined_weight)
                out.append(conn)

            j = j + 1
        i = i + 1
    #sys.stdout.write('out length: %d' % len(out))
    return out

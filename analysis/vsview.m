% Visualise several VentralStream plots at once.
%
% Usage: vsview (data, t)
function vsview (data, t)

    vssurf(1, data, 'World', t, 1)
    vssetgrid ([-1 2]);

    vssurf(2, data, 'V1_p_edges', t, 1)
    vssetgrid ([0 2]);
    vssurf(3, data, 'V1_r_edges', t, 1)
    vssetgrid ([1 2]);

    vssurf(4, data, 'V2_p_lines', t, 1)
    vssetgrid ([2 2]);
    vssurf(5, data, 'V2_r_lines', t, 1)
    vssetgrid ([3 2]);

    vssurf(6, data, 'V2_sw', t, 1)
    vssetgrid ([-1 1]);

end
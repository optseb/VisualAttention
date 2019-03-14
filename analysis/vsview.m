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

    vssurf(6, data, 'V2_pPp_rPr', t, 1)
    vssetgrid ([-1 1]);

    vssurf(7, data, 'V2_pPp_rMr', t, 1)
    vssetgrid ([0 1]);

    vssurf(8, data, 'V2_pMp_rPr', t, 1)
    vssetgrid ([1 1]);

    vssurf(9, data, 'V2_pMp_rMr', t, 1)
    vssetgrid ([2 1]);

end
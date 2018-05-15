% Plot a surface plot with some standard views.
function vssurf (fign, data, field, time, surf_view=1, toplim=1, botlim=0)
    figure(fign)
    surf (data.(field)(:,:,time))
    title (field)
    zlim([botlim, toplim])
    % Add some useful views here:
    if (surf_view==1)
        view([1,85])
    elseif (surf_view==2) % To be good for a negative hump
        view([30,10])
    else
        view([20,89])
    end
end

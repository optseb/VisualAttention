% Plot a surface plot with some standard views.
%function vssurf (fign, data, field, time, surf_view=1, toplim=1,
%botlim=0)
function vssurf (fign, data, field, time, surf_view, toplim, botlim)

    % Note, default values for args is octave only. This should be portable
    if nargin < 5
        surf_view = 1;
    end
    if nargin < 6
        toplim = 1
    end
    if nargin < 7
        botlim = 0
    end

    figure(fign)
    surf (data.(field)(:,:,time))
    % Escape underscores and set title:
    titlestr = regexprep (field, '_', '\_');
    title (titlestr);
    thelims = [botlim toplim]
    zlim(thelims)
    % Add some useful views here:
    if (surf_view==1)
        view([82,86])
    elseif (surf_view==2) % To be good for a negative hump
        view([30,10])
    else
        view([20,89])
    end
end

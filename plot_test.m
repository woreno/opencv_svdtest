function plot_test()

clc, closeall

function plt_(title_, x_, d1, d2, lg1, lg2,xAxisLg) 
    h1 = plot(x,d1,'color','blue','linewidth', 2);
    hold on;
    h2 = plot(x,d2,'color','red','linewidth', 2);

    h_ = legend ([h1,h2],lg1,lg2); set(h_, 'fontsize', 12);
    xlabel(xAxisLg);
    ylabel('microseconds');
    title(title_);
end
 

% append here the results


for i=1:4
    set(figure(i), 'position', [1   524  1922   447]);
end

end
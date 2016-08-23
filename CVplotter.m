%CVplotter

%cd a12tau01
%{
for jj = 1:2
    figure(jj); %set(gcf,'Position',get(0,'Screensize'))
    subplot(2,5,1); plotImage(data(:,:,1,jj)); c = caxis;
    title('Input'); colorbar('location','southoutside')
    for ii = 1:9
        subplot(2,5,ii+1); plotImage(x(:,:,ii,jj)); caxis(c);
        title(sprintf('tau = %2.3f',tau(ii)))
    end
end
clear c ii jj
%figure(1); saveas(gcf,'Det1.jpg');
%figure(2); saveas(gcf,'Det2.jpg');
%}

%
%col = 43; % a 
col = 35; % b
for jj = 1:2
    figure(jj+2);
    plot(data(:,col,1,jj),'LineWidth',2); hold on; 
    for ii = 1:9
        plot(x(:,col,ii,jj));
    end
    hold off
    legend('Input','10','1','0.5','0.2','0.1','0.05','0.02','0.01','0.001'...
        ,'location','best');
    title(sprintf('Column %d',col));
end
clear ii jj 
%figure(3); saveas(gcf,'Det1_prof.jpg');
%figure(4); saveas(gcf,'Det2_prof.jpg');
%}

%cd ..
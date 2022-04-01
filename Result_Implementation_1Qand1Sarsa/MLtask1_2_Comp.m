close all
clear all
clc

%% Load Data
folderQ = uigetdir;
%% 
listingQ=dir(folderQ);
for i = 3:length(listingQ)
    round = listingQ(i).name;
    dataQ{i-2} = readmatrix([folderQ '\' round]);
end
dataQQ = [dataQ(1); dataQ(2); dataQ(3);dataQ(4);dataQ(5);dataQ(6)];
dataQ = cell2mat(dataQQ);
dataQ = dataQ(:,[2,4]);
for i = 1:5
    meanQ(i) = mean(dataQ(1+50*(i-1):i*50,1));%step spend
    stdQ(i) = std(dataQ(1+50*(i-1):i*50,1));%step spend
    meanQC(i) = mean(dataQ(1+50*(i-1):i*50,2));%coin left
    stdQC(i) = std(dataQ(1+50*(i-1):i*50,2));%coin left
    i = i+1;
end
meanQ(6) = mean(dataQ(251:450,1));
meanQC(6) = mean(dataQ(251:450,2));
stdQ(6) = std(dataQ(251:450,1));
stdQC(6) = std(dataQ(251:450,2));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5
%% Load Data
folderQS = uigetdir;
%% 
listingQS=dir(folderQS);
for i = 3:length(listingQS)
    round = listingQS(i).name;
    dataQS{i-2} = readmatrix([folderQS '\' round]);
end
dataQQS = [dataQS(1); dataQS(2); dataQS(3);dataQS(4);dataQS(5);dataQS(6)];
dataQS = cell2mat(dataQQS);
dataQS = dataQS(:,[2,4]);
for i = 1:5
    meanQS(i) = mean(dataQS(1+50*(i-1):i*50,1));%step spend
    stdQS(i) = std(dataQS(1+50*(i-1):i*50,1));%step spend
    meanQCS(i) = mean(dataQS(1+50*(i-1):i*50,2));%coin left
    stdQCS(i) = std(dataQS(1+50*(i-1):i*50,2));%coin left
    i = i+1;
end
meanQS(6) = mean(dataQS(251:450,1));
meanQCS(6) = mean(dataQS(251:450,2));
stdQS(6) = std(dataQS(251:450,1));
stdQCS(6) = std(dataQS(251:450,2));

%plot step spend%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure()
hold on
%---10----------
x1 = [0 50 50 0];
y1 = [-5 -5 450 450];
s1 = fill(x1,y1,'red');
s1.FaceAlpha = 0.1;
%-----50------------
x2 = [50 100 100 50];
y2 = [-5 -5 450 450];
s2 = fill(x2,y2,'magenta');
s2.FaceAlpha = 0.1;
%-----100------------
x3 = [100 150 150 100];
y3 = [-5 -5 450 450];
s3 = fill(x3,y3,'blue');
s3.FaceAlpha = 0.1;
%-----200------------
x4 = [150 200 200 150];
y4 = [-5 -5 450 450];
s4 = fill(x4,y4,'cyan');
s4.FaceAlpha = 0.1;
%-----500------------
x5 = [200 250 250 200];
y5 = [-5 -5 450 450];
s5 = fill(x5,y5,'green');
s5.FaceAlpha = 0.1;
%-------------------
x6 = [250 450 450 250];
y6 = [-5 -5 450 450];
s6 = fill(x6,y6,'yellow');
s6.FaceAlpha = 0.1;
xstd=[25;75;125;175;225;350];
%xstdC=[10;50;100;200;500;1000];
errorbar(xstd,meanQ',stdQ','-s','MarkerSize',6,...
    'MarkerEdgeColor',[0 0.4470 0.7410],'MarkerFaceColor',[0 0.4470 0.7410], ...
    'Color',[0 0.4470 0.7410],'LineWidth',1.5,'CapSize', 10)
errorbar(xstd,meanQS',stdQS','-s','MarkerSize',6,...
    'MarkerEdgeColor',[0.8500 0.3250 0.0980],'MarkerFaceColor',[0.8500 0.3250 0.0980], ...
    'Color',[0.8500 0.3250 0.0980],'LineWidth',1.5,'CapSize', 10)
ylim([100,450])
xlabel('round')
ylabel('steps of spending')
title('Steps to Collect Coins(Mean~Std)')
legend('10','50','100','200','500','1000','Q-leaning','SARSA')
%% plot coin left#########################################################
figure()
hold on
x1 = [0 50 50 0];
y1 = [-5 -5 400 400];
s1 = fill(x1,y1,'red');
s1.FaceAlpha = 0.1;
%-----50------------
x2 = [50 100 100 50];
y2 = [-5 -5 400 400];
s2 = fill(x2,y2,'magenta');
s2.FaceAlpha = 0.1;
%-----100------------
x3 = [100 150 150 100];
y3 = [-5 -5 400 400];
s3 = fill(x3,y3,'blue');
s3.FaceAlpha = 0.1;
%-----200------------
x4 = [150 200 200 150];
y4 = [-5 -5 400 400];
s4 = fill(x4,y4,'cyan');
s4.FaceAlpha = 0.1;
%-----500------------
x5 = [200 250 250 200];
y5 = [-5 -5 400 400];
s5 = fill(x5,y5,'green');
s5.FaceAlpha = 0.1;
%-------------------
x6 = [250 450 450 250];
y6 = [-5 -5 400 400];
s6 = fill(x6,y6,'yellow');
s6.FaceAlpha = 0.1;
xstdC=[25;75;125;175;225;350];
%xstdC=[10;50;100;200;500;1000];
errorbar(xstdC,meanQC',stdQC','-s','MarkerSize',6,...
    'MarkerEdgeColor',[0 0.4470 0.7410],'MarkerFaceColor',[0 0.4470 0.7410], ...
    'Color',[0 0.4470 0.7410],'LineWidth',1.5,'CapSize', 10)
errorbar(xstdC,meanQCS',stdQCS','-s','MarkerSize',6,...
    'MarkerEdgeColor',[0.8500 0.3250 0.0980],'MarkerFaceColor',[0.8500 0.3250 0.0980], ...
    'Color',[0.8500 0.3250 0.0980],'LineWidth',1.5,'CapSize', 10)
ylim([-5,30])
xlabel('round')
ylabel('Coins Left')
title('Coins Left(Mean~Std)')
legend('10','50','100','200','500','1000','Q-leaning','SARSA')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

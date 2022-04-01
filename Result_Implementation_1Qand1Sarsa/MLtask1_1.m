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
    meanQC(i) = mean(dataQ(1+50*(i-1):i*50,2));%coin left
    i = i+1;
end
meanQ(6) = mean(dataQ(251:450,1));
meanQC(6) = mean(dataQ(251:450,2));

%plot step spend%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure()
hold on
%-----10------------
x1 = [0 50 50 0];
y1 = [0 0 400 400];
s1 = fill(x1,y1,'red');
s1.FaceAlpha = 0.1;
%-----50------------
x2 = [50 100 100 50];
y2 = [0 0 400 400];
s2 = fill(x2,y2,'magenta');
s2.FaceAlpha = 0.1;
%-----100------------
x3 = [100 150 150 100];
y3 = [0 0 400 400];
s3 = fill(x3,y3,'blue');
s3.FaceAlpha = 0.1;
%-----200------------
x4 = [150 200 200 150];
y4 = [0 0 400 400];
s4 = fill(x4,y4,'cyan');
s4.FaceAlpha = 0.1;
%-----500------------
x5 = [200 250 250 200];
y5 = [0 0 400 400];
s5 = fill(x5,y5,'green');
s5.FaceAlpha = 0.1;
%-------------------
x6 = [250 450 450 250];
y6 = [0 0 400 400];
s6 = fill(x6,y6,'yellow');
s6.FaceAlpha = 0.1;
plot(dataQ(:,1),'LineWidth',1,'color','black')
for i = 1:5
    plot([0+50*(i-1),i*50],[meanQ(i),meanQ(i)],'LineWidth',3,'Color','r');
end
plot([250,450],[meanQ(6),meanQ(6)],'LineWidth',3,'Color','r');
legend('10','50','100','200','500','1000','Steps','Mean')
xlabel('round')
ylabel('steps of spending')
title('Steps to Collect Coins')
%% plot coin left#########################################################
figure()
hold on
%-----10------------
x1 = [0 50 50 0];
y1 = [0 0 50 50];
s1 = fill(x1,y1,'red');
s1.FaceAlpha = 0.1;
%-----50------------
x2 = [50 100 100 50];
y2 = [0 0 50 50];
s2 = fill(x2,y2,'magenta');
s2.FaceAlpha = 0.1;
%-----100------------
x3 = [100 150 150 100];
y3 = [0 0 50 50];
s3 = fill(x3,y3,'blue');
s3.FaceAlpha = 0.1;
%-----200------------
x4 = [150 200 200 150];
y4 = [0 0 50 50];
s4 = fill(x4,y4,'cyan');
s4.FaceAlpha = 0.1;
%-----500------------
x5 = [200 250 250 200];
y5 = [0 0 50 50];
s5 = fill(x5,y5,'green');
s5.FaceAlpha = 0.1;
%-------------------
x6 = [250 450 450 250];
y6 = [0 0 50 50];
s6 = fill(x6,y6,'yellow');
s6.FaceAlpha = 0.1;
plot(dataQ(:,2),'LineWidth',1,'color','black')
for i = 1:5
    plot([0+50*(i-1),i*50],[meanQC(i),meanQC(i)],'LineWidth',3,'Color','r');
end
plot([250,450],[meanQC(6),meanQC(6)],'LineWidth',3,'Color','r');
legend('10','50','100','200','500','1000','Coins','Mean')
xlabel('round')
ylabel('Coins Left')
title('Coins Left')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

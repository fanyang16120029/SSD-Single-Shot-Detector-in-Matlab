%% 
%-------------SSD_Video_Script-----------
%作  者：杨帆
%公  司：BJTU
%功  能：SSD模拟程序(for pic)。
%输  入：
%       Video_Path      -----> 输入图像路径。
%       Description     -----> 参数结构体。
%输  出：
%       
%备  注：Matlab 2016a。
%----------------------------------------

%%
% 清空工作空间

clear all;
clc

%%
% 初始参数设定

Video_Path = 'E:\2017_12_26_ 行人检测本地数据集\JPEG\';
Ground_Truth_Path = 'E:\2017_12_26_ 行人检测本地数据集\Mat\';

Description.aspect_ratio(1).r = [2, 1/2];
Description.aspect_ratio(2).r = [2, 1/2, 3, 1/3];
Description.aspect_ratio(3).r = [2, 1/2, 3, 1/3];
Description.aspect_ratio(4).r = [2, 1/2, 3, 1/3];
Description.aspect_ratio(5).r = [2, 1/2];
Description.aspect_ratio(6).r = [2, 1/2];

Description.feature_size = [38, 38; 19, 19; 10, 10; 5, 5; 3, 3; 1, 1];
Description.scale = [0.15, 0.2, 0.37, 0.54, 0.71, 0.88];

%% 
% 网络加载。

net = Load_Net();

%%
% 视频序列读取。

Video_Dir = dir(Video_Path);  
Video_Len = size(Video_Dir, 1); 
Detect_Result = {};
for i = 3: Video_Len  
    Img_Name = strcat(Video_Path, Video_Dir(i).name);
    Img = imread(Img_Name);
    [height, width, depth] = size(Img);
    Roi_Table = SSD_Net(net, Img, 21, Description);
    if(~isempty(Roi_Table))
        Roi_Table = [Roi_Table(:,1), Roi_Table(:,2), width * Roi_Table(:,3)...
        height * Roi_Table(:,4), width * Roi_Table(:,5), height * Roi_Table(:,6)];
        Roi_Pick = find(Roi_Table(:,1) == 16);
        Roi_Table = Roi_Table(Roi_Pick, :);
    end
    Roi_Pick = NMS(Roi_Table, 0.45, 'NULL');  
    if(~isempty(Roi_Table))
        Detect_Result{i - 2} = [Roi_Table(Roi_Pick, 3: 4),...
            Roi_Table(Roi_Pick, 5) - Roi_Table(Roi_Pick, 3),...
            Roi_Table(Roi_Pick, 6) - Roi_Table(Roi_Pick, 4), ...
            Roi_Table(Roi_Pick, 2)];
    else
        Detect_Result{i - 2} = [];
    end
    disp(strcat('Frame ', int2str(i - 2), ' done.'));
    
    Ground_Truth_Name = strcat(Ground_Truth_Path, Video_Dir(i).name(1:end - 4), '.mat');
    load(Ground_Truth_Name);
    Ground_Truth{i - 2} = labelingSession.ImageSet.ImageStruct.objectBoundingBoxes  ; 
    Ground_Truth{i - 2} = [Ground_Truth{i - 2}, zeros(size(Ground_Truth{i - 2}, 1), 1)];
%     result_img = Img;
% 
%     for j = 1: size(Detect_Result{i - 2}, 1)
%         rois = Detect_Result{i - 2};
%         left_x = max(round(rois(j, 1)), 1);
%         left_y = max(round(rois(j, 2)), 1);
%         width = round(rois(j, 3));
%         height = round(rois(j, 4));
% 
%         result_img = drawRect( result_img, [left_x, left_y], ...
%             [width, height], 3, [0, 255, 0]);
%         imshow(result_img);
%     end
end

ref = 10.^(-2:.25:0);
lims = [3.1e-3 1e1 .05 1];
[gt,dt] = bbGt('evalRes',Ground_Truth,Detect_Result,0.5,0);
[fp,tp,score,miss,pre] = bbGt('compRoc', gt, dt, 1, ref);
miss=exp(mean(log(max(1e-10,1-miss))));
figure; 
plotRoc([fp tp],'logx',1,'logy',1,'xLbl','fppi',...
  'lims',lims,'color','g','smooth',1,'fpTarget',ref);
title(sprintf('log-average miss rate = %.2f%%',miss*100));
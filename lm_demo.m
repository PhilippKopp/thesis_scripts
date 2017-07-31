%% demo code for facial landmark detection
close all
clear
clc

addpath('/user/HS204/m09113/facer2vm_project_area/people/Philipp/Zhenhua_lms_DAC_CSR/')
run /user/HS204/m09113/facer2vm_project_area/Share/DAC_CSR/vlfeat-0.9.20/toolbox/vl_setup.m;
%img_list = dir('/user/HS204/m09113/facer2vm_project_area/Share/DAC_CSR/data/*.png');
fid = fopen('/user/HS204/m09113/facer2vm_project_area/data/300VW_Dataset_2015_12_14/bb_clicked_philipp.log','r');
C = textscan(fid, '%s %d %d %d %d', 'Delimiter', ' ');
first_img_list = C{1};
%class(img_list)


bboxs = double(cell2mat(C(1:end,2:5)));
bboxs (1:end,3) = bboxs(1:end,3)-bboxs(1:end,1);
bboxs (1:end,4) = bboxs(1:end,4)-bboxs(1:end,2);

load /user/HS204/m09113/facer2vm_project_area/Share/DAC_CSR/cr_model_68.mat;

show_results = false;
write_pts = true;

%global frame_num
faceDetector = vision.CascadeObjectDetector();
disp('created facedetector')

parfor i = 1:length(first_img_list)
    pre_bb = [ 0 0 0 0 ];
    % load image
    %img = imread(['/user/HS204/m09113/facer2vm_project_area/Share/DAC_CSR/data/', img_list(i).name]);
    %class(img_list{i})
    first_img_list{i}(1:end-10)
    all_vid_frames =  dir([first_img_list{i}(1:end-10) '*.png']);
    if write_pts
        mkdir ([first_img_list{i}(1:end-17) 'CSR_lms']);
    end
    
    for frame_num = 1:length(all_vid_frames)
        disp(all_vid_frames(frame_num).name);
        %try
            %img = imread(first_img_list{i});
            img = imread([first_img_list{i}(1:end-10) all_vid_frames(frame_num).name]);
            
            
            if frame_num==1
                bbox = bboxs(i,1:end);
            else
                if pre_bb(3) > 20 && pre_bb(4) > 20
                    bbox = pre_bb;
                else
                    % Create a cascade detector object.
                    bbox = step(faceDetector, img);
                    if size(bbox,1) > 0
                        bbox = bbox(1,1:end);
                    else
                        bbox = pre_bb;
                    end
                end
            end

            % load bbox and ground truth facial landmarks
            %load(['/user/HS204/m09113/facer2vm_project_area/Share/DAC_CSR/data/', img_list(i).name(1:end-3), 'mat']);
            %gt_lmk = lmk_bbox.gt_lmk;
            %bbox = lmk_bbox.bbox;

            % landmark initalisation
            init_lmk = project_s2b(cr_model.mean_shape, bbox);
            
            %if frame_num > 580
            %    init_lmk
            %end

            
            % face landmarking
            pre_lmk = fit_sdt(rgb2gray(img), init_lmk, cr_model);
            
            % calc bb around predicted lms, that can be used as init for next
            % frame
            pre_bb = [ 0 0 0 0 ];
            pre_bb(1)=min(pre_lmk(1:end/2));
            pre_bb(2)=min(pre_lmk(end/2+1:end));
            pre_bb(3)=max(pre_lmk(1:end/2)) - pre_bb(1);
            pre_bb(4)=max(pre_lmk(end/2+1:end)) - pre_bb(2);

            % display result
            if show_results && frame_num>550
                imshow(img);
                hold on;
            
                %plot(gt_lmk(1:end/2), gt_lmk(end/2+1:end), 'yo', 'markerfacecolor', 'y');
                plot(pre_lmk(1:end/2), pre_lmk(end/2+1:end), 'ro', 'markerfacecolor', 'r');
            
                hold off;
                title('press any key for the next image');
                pause();
            end
            
            % write lm files
            if write_pts
                ofid = fopen([first_img_list{i}(1:end-17) 'CSR_lms/' all_vid_frames(frame_num).name(1:end-3) 'pts'], 'w');
                fprintf(ofid,'version: 1\n');
                fprintf(ofid,'n_points: 68\n');
                fprintf(ofid,'{\n');
                num_lms=length(pre_lmk(1:end/2));
                for lmi = 1:num_lms
                    fprintf(ofid,'%f %f\n',pre_lmk(lmi),pre_lmk(lmi+num_lms));
                end
                fprintf(ofid,'}\n');
                fclose(ofid);
            end
        %catch
        %    err = lasterror;
        %    disp(err);
        %    disp(err.message);
        %    disp(err.stack);
        %    disp(err.identifier);
        %end
    end
end
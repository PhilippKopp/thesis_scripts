%% demo code for facial landmark detection
close all
clear
clc

addpath('/user/HS204/m09113/facer2vm_project_area/people/Philipp/Zhenhua_lms_DAC_CSR/')
run /user/HS204/m09113/facer2vm_project_area/Share/DAC_CSR/vlfeat-0.9.20/toolbox/vl_setup.m;
%img_list = dir('/user/HS204/m09113/facer2vm_project_area/Share/DAC_CSR/data/*.png');
fid = fopen('/user/HS204/m09113/facer2vm_project_area/data/300VW_Dataset_2015_12_14/bb_clicked_philipp_menpo.log','r');
C = textscan(fid, '%s %d %d %d %d', 'Delimiter', ' ');
first_img_list = C{1};
fclose(fid);
%class(img_list)


bboxs = double(cell2mat(C(1:end,2:5)));
bboxs (1:end,3) = bboxs(1:end,3)-bboxs(1:end,1);
bboxs (1:end,4) = bboxs(1:end,4)-bboxs(1:end,2);

load /user/HS204/m09113/facer2vm_project_area/Share/DAC_CSR/cr_model_68.mat;

show_results = false;
write_pts = true;

%global frame_num

bb_log_path = '/user/HS204/m09113/my_project_folder/menpo_challenge/300vw_trainingsset_fittings/face_detect_fasterRCNN/conf_0.80/VGG16_expr_init_JointTraning_WF_PRN_MIN_SIZE_8_allBB/';

parfor i = 1:length(first_img_list)
    pre_bb = [ 0 0 0 0 ];
    pre_lmk = zeros(136);
    % load image
    %img = imread(['/user/HS204/m09113/facer2vm_project_area/Share/DAC_CSR/data/', img_list(i).name]);
    %class(img_list{i})
    first_img_list{i}(1:end-10)
    all_vid_frames =  dir([first_img_list{i}(1:end-10) '*.png']);
    if write_pts
        mkdir ([first_img_list{i}(1:end-17) 'CSR_lms_rcnnBB']);
    end
    
    for frame_num = 1:length(all_vid_frames)
        disp(all_vid_frames(frame_num).name);
        %try
            %img = imread(first_img_list{i});
            img = imread([first_img_list{i}(1:end-10) all_vid_frames(frame_num).name]);
            
            %read bb provided by rcnn
            bb_file_path = [bb_log_path first_img_list{i}(end-21:end-18) '/' all_vid_frames(frame_num).name(1:end-4) '.txt'];
            fid = fopen(bb_file_path, 'r');
            rcnn_bb = textscan(fid, '%f %f %f %f %f %f', 'Delimiter', ' ');
            %nrows = numel(cell2mat(textscan(fid,'\n')))
            %nrows = numel(textscan(fid,'%1c%*[^\n]'))
            bb_provided = ~isempty(rcnn_bb{1});
            if bb_provided
                rcnn_bb = cell2mat(rcnn_bb);
                rcnn_bb(:,3) = rcnn_bb(:,3) - rcnn_bb(:,1);
                rcnn_bb(:,4) = rcnn_bb(:,4) - rcnn_bb(:,2);
                argmax = 1;
                %[max ,argmax] = max(rcnn_bb(:,6))
                if size(rcnn_bb,1)>1
                    rcnn_bb(:,6) = rcnn_bb(:,3).*rcnn_bb(:,4); %size of facebox as qual measurement
                    [max_val, argmax] = max(rcnn_bb(:,6));
                end
                rcnn_bb = rcnn_bb(argmax,1:4);
                %disp (rcnn_bb);
            end
            fclose(fid);
            
            if frame_num==1
                bbox = bboxs(i,1:end);
                init_lmk = project_s2b(cr_model.mean_shape, bbox);
            else
                if bb_provided
                    % if landmarks of previous frame in bb of this frame
                    % left bb border < left_eye_x(37) AND right bb border >
                    % right_eye_x(46) AND upper bb border < bridge_y(28) AND
                    % lower bb boder > lower_lip_y(58)
                    if rcnn_bb(1) < pre_lmk(37) && rcnn_bb(1)+rcnn_bb(3) > pre_lmk(46) && rcnn_bb(2) < pre_lmk(68+28) && rcnn_bb(2)+rcnn_bb(4) > pre_lmk(68+58)
                        %bbox = pre_bb;
                        init_lmk = pre_lmk;
                    else
                        disp('reinitialized landmarks');
                        %cut upper 25% of the bb
                        rcnn_bb(2) = rcnn_bb(2) + 0.25 *rcnn_bb(4);
                        rcnn_bb(4) = 0.75 *rcnn_bb(4);
                        init_lmk = project_s2b(cr_model.mean_shape, rcnn_bb);
                        %show_results=false;
                    end
                else
                    init_lmk = pre_lmk;
                end
            end

            % load bbox and ground truth facial landmarks
            %load(['/user/HS204/m09113/facer2vm_project_area/Share/DAC_CSR/data/', img_list(i).name(1:end-3), 'mat']);
            %gt_lmk = lmk_bbox.gt_lmk;
            %bbox = lmk_bbox.bbox;

            % landmark initalisation
            %init_lmk = project_s2b(cr_model.mean_shape, bbox);
            
            %if frame_num > 580
            %    init_lmk
            %end

            
            % face landmarking
            pre_lmk = fit_sdt(rgb2gray(img), init_lmk, cr_model);
            
            % calc bb around predicted lms, that can be used as init for next
            % frame
            %pre_bb = [ 0 0 0 0 ];
            %pre_bb(1)=min(pre_lmk(1:end/2));
            %pre_bb(2)=min(pre_lmk(end/2+1:end));
            %pre_bb(3)=max(pre_lmk(1:end/2)) - pre_bb(1);
            %pre_bb(4)=max(pre_lmk(end/2+1:end)) - pre_bb(2);

            % display result
            if show_results
                imshow(img);
                hold on;
            
                %plot(gt_lmk(1:end/2), gt_lmk(end/2+1:end), 'yo', 'markerfacecolor', 'y');
                plot(pre_lmk(1:end/2), pre_lmk(end/2+1:end), 'ro', 'markerfacecolor', 'r');
            
                hold off;
                title('press any key for the next image');
                %pause();
                drawnow;
                %show_results=false;
            end
            
            % write lm files
            if write_pts
                ofid = fopen([first_img_list{i}(1:end-17) 'CSR_lms_rcnnBB/' all_vid_frames(frame_num).name(1:end-3) 'pts'], 'w');
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

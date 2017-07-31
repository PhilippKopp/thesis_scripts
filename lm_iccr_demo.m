close all
clear
clc

addpath('/user/HS204/m09113/my_project_folder/menpo_challenge/300vw_emax_iccr_lms/iCCR')

load model.mat 
load example_params.mat 
params.show = 0;
%video = '/user/HS204/m09113/facer2vm_project_area/data/300VW_Dataset_2015_12_14/112/vid.avi';
video_base = '/user/HS204/m09113/facer2vm_project_area/data/300VW_Dataset_2015_12_14/';
videos = dir([video_base '*']);
%length(videos)
%return
for i = 103:124
    if videos(i).isdir
        if videos(i).name=='.'
            continue
        end
        video = [video_base videos(i).name '/vid.avi'];
        disp(video)
        try
            data = track( model, video, params );
            out_path = [video_base videos(i).name '/iccr_lms'];
            mkdir(out_path);
            out_path = [out_path '/all_lms.mat'];
            save(out_path, 'data');
            %iSaveX(out_path, data);
            %return
        catch
        end
    end
end    




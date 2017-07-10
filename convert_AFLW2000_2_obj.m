
%% Load Model
data_path = '/user/HS204/m09113/my_project_folder/AFLW2000_fittings/3DDFA/inputs/iter3/';
load('Model_Shape.mat');
load('Model_Exp.mat');
mu = mu_shape + mu_exp;

%% convert 
%filelist = dir ('/user/HS204/m09113/facer2vm_project_area/data/AFLW2000-3D/*.mat');
filelist = dir ([data_path '*.mat']);
for fi = 1:length(filelist);
    fi
    sample_name = filelist(fi).name(1:end-4);
    if exist(['/user/HS204/m09113/my_project_folder/AFLW2000_fittings/3DDFA/iter3_converted2obj/' sample_name(6:end) '.obj'], 'file')==2
        disp(['already finished ' sample_name]);
        continue
    end
    
    % copied from main_show_with_BFM.m
    %img = imread([data_path sample_name '.jpg']);
    load([data_path sample_name '.mat']);
    %[height, width, nChannels] = size(img);
    
    % only necessary when converting 3DDFA fitting result:
    Shape_Para = para0(7+1:7+199);
    Exp_Para   = para0(7+199+1:7+199+29);

    vertex = mu + w * Shape_Para + w_exp * Exp_Para;
    vertex = reshape(vertex, 3, length(vertex)/3);
    %tex = mu_tex + w_tex * Tex_Para;
    %tex = reshape(tex, 3, length(mu_tex)/3);
    norm = NormDirection(vertex, tri);

    %write_obj(['/user/HS204/m09113/facer2vm_project_area/data/AFLW2000-3D/converted2obj/' sample_name '.obj'], vertex, tri);
    write_obj(['/user/HS204/m09113/my_project_folder/AFLW2000_fittings/3DDFA/iter3_converted2obj/' sample_name(6:end) '.obj'], vertex, tri);
end



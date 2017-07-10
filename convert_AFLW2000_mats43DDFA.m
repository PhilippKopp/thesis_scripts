path = '/user/HS204/m09113/facer2vm_project_area/data/AFLW2000-3D/';
filelist = dir ([path '*.mat']);
for fi = 1:length(filelist);
    file_name = filelist(fi).name
    load([path file_name])
    pt2d_3d = pt3d_68(1:2,1:end);
    save([path file_name(1:end-4) '_3DDFA.mat'], 'pt2d', 'pt2d_3d')
end

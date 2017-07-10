function [ done ] = export_images_and_lms( landmarkfile, exportfolder )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

old_landmarks_used=0;
mkdir(exportfolder)
for i=1:1:size(landmarkfile, 2)
    
    i
    
    [pathstr,name,ext] = fileparts(landmarkfile{i}.file);
    %if strcmp(landmarkfile{i}.nfile, 'img/36600.JPG')
    if strcmp(name, '2736')
        landmarkfile{i}
        return
    end
    
    has_new_landmarks = isfield(landmarkfile{i}, 'landmark68_new');
    has_old_landmarks = isfield(landmarkfile{i}, 'landmark68');
    if (~(has_new_landmarks || has_old_landmarks))
        continue; % no landmarks, skip image
    end
    
    if (~has_new_landmarks && has_old_landmarks)
        old_landmarks_used=old_landmarks_used+1
    end
    
    if exist(strcat( exportfolder, int2str(landmarkfile{i}.id), '/', name, '.pts'),'file')==2
        continue;
    end
    strcat( exportfolder, int2str(landmarkfile{i}.id), '/', name, '.pts')
    %Disp('okay we have a problem')
    
    if ~exist(strcat(exportfolder, int2str(landmarkfile{i}.id)), 'dir')
        mkdir(strcat(exportfolder, int2str(landmarkfile{i}.id)));
    end
    
    %assemble link command
    cmd = {'ln -s '};
    %cmd = string(cmd);
    cmd= strcat( cmd , landmarkfile{i}.file );
    cmd= strcat( cmd, {' '}, exportfolder , int2str(landmarkfile{i}.id) ,'/' , name , ext);
    system(sprintf('%s',cmd{1}));
    
    if has_new_landmarks
        write_lms(landmarkfile{i}.landmark68_new, strcat( exportfolder, int2str(landmarkfile{i}.id), '/', name, '.pts'));
    elseif has_old_landmarks
        write_lms(landmarkfile{i}.landmark68, strcat( exportfolder, int2str(landmarkfile{i}.id), '/', name, '.pts'));
    end
      
end

end


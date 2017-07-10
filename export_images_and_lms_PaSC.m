function [ done ] = export_images_and_lms( landmarkfile, exportfolder )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

DB_BASE='/vol/vssp/datasets/multiview/PaSC/stills/';
mkdir(exportfolder)
for i=1:1:size(landmarkfile, 2)
%for i=1:1:10
    
    i
    
    has_landmarks = isfield(landmarkfile(i), 'landmark');
    if (~has_landmarks)
        continue; % no landmarks, skip image
    end
    
    %[pathstr,name,ext] = fileparts(landmarkfile(i).frame_id{1});
    [pathstr,name,ext] = fileparts(landmarkfile(i).image_id);
    %pathstr
    %name
    
    id = strsplit(landmarkfile(i).image_id,'d');
    id = id {1};
    %return 
    
    if ~exist(strcat(exportfolder, id), 'dir')
        mkdir(strcat(exportfolder, id));
    end
    
    
    %assemble link command
    cmd = {'ln -s '};
    %cmd = string(cmd);
    %cmd= strcat( cmd , DB_BASE, landmarkfile(i).frame_id{1} );
    cmd= strcat( cmd , DB_BASE, landmarkfile(i).image_id);
    %cmd= strcat( cmd, {' '}, exportfolder , pathstr ,'/' , name , ext);
    cmd= strcat( cmd, {' '}, exportfolder, id ,'/' , name , ext);
    cmd= strcat( cmd, {' &'});

    system(sprintf('%s',cmd{1}));
    %write_lms(landmarkfile(i).landmark, strcat( exportfolder, pathstr, '/', name, '.pts'));
    write_lms(landmarkfile(i).landmark, strcat( exportfolder, id, '/', name, '.pts'));
    
    %fclose('all') % close all open files
      
end

end


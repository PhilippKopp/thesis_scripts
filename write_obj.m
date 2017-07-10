function [ output_args ] = write_obj( fullfilename, vertices, tris )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
fid = fopen(fullfilename,'w');

%for i=1:length(vertices)
for i=1:size(vertices,2)
    fprintf(fid,'v %5f %5f %5f\n', vertices(1,i), vertices(2,i), vertices(3,i));
end

for i=1:size(tris,2)
    fprintf(fid,'f %5.5f %5.5f %5.5f\n', tris(1,i), tris(2,i), tris(3,i));
end

fclose(fid);
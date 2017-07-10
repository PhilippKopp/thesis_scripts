function [ done ] = write_lms( lms_as_list, output_file )
%EXPORT_ Summary of this function goes here
%   Detailed explanation goes here

%print header
fileID = fopen(output_file,'wt', 'n','UTF-8');
fprintf(fileID,'version: 1\n');
fprintf(fileID,'n_points: 68\n');
fprintf(fileID,'{\n');

number_of_lms = size(lms_as_list);
number_of_lms = number_of_lms(1)/2;

for n = 1:number_of_lms
    fprintf(fileID,'%f %f \n',lms_as_list(n), lms_as_list(number_of_lms + n));
end
fprintf(fileID,'}\n');

fclose(fileID);

end


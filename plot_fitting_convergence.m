%close all;
%data = csvread('D:\Github\experiments\02122016-fitting-convergence\300VW_002_000008_edge_pose_shpexpr.csv');
%data = csvread('D:\Github\experiments\02122016-fitting-convergence\300VW_002_000008_edge_pose_shp_expr.csv');
%data = csvread('D:\Github\experiments\02122016-fitting-convergence\300VW_002_000008_cnt_edgecnt_pose_shp_expr.csv');
%data = csvread('D:\Github\experiments\02122016-fitting-convergence\300VW_002_000008_cnt_edgecnt_pose_shp_expr__some_initialisation_fixes.csv');
%data = csvread('D:\Github\experiments\02122016-fitting-convergence\300VW_002_000008_cnt_edgecnt_pose_shp_expr__some_initialisation_fixes__cnt_fix.csv');
%data = csvread('D:\Github\experiments\02122016-fitting-convergence\300VW_002_000008_fit_shape_and_pose_eos-v0.9.1.csv'); % lambda = 30. This should be more or less the same as one above, except maybe different regularisation.
%data = csvread('D:\Github\experiments\02122016-fitting-convergence\300VW_002_000008_fit_shape_and_pose_eos-v0.9.1_10_shp_coeffs.csv'); % lambda = 30, 10 shp coeffs fitted
%data = csvread('D:\Github\experiments\02122016-fitting-convergence\300VW_002_000008_fit_shape_and_pose_eos-v0.12.0-alpha (eigen).csv'); % lambda = 30, 10 shp coeffs fitted.
data = csvread('/user/HS204/m09113/my_project_folder/multi_fitting_convergence_tests/KF-ITW_multi_02_happy_without_bad_conv.csv');
%data = csvread('/user/HS204/m09113/out_2_multi_solve_400.csv'); %single example image
%data = csvread('/user/HS204/m09113/out.csv');
% I think they're all generated with the 845r model. At least the eos-v0.9.1 ones are.
% Same for the num_shape_coeffs_to_fit, I used all in the v0.9.1 ones.
num_iter = size(data, 1) / 3;
num_shp = 10;
num_bs = 6;
num_pose = 6;

plot_iter = num_iter;
pose = data(1:3:end, 1:num_pose);
shp = data(2:3:end, 1:num_shp);
bs = data(3:3:end, 1:num_bs);

figure(1);
subplot(1, 3, 1); plot(pose(1:plot_iter,1:3)); title('Rotation (Y, P, R)'); grid on; xlim([0 num_iter]); 
subplot(1, 3, 2); plot(pose(1:plot_iter,4:5)); title('Translation'); grid on; xlim([0 num_iter]);
subplot(1, 3, 3); plot(pose(1:plot_iter,6)); title('Scale'); grid on; xlim([0 num_iter]);
saveas(1,'fig1.png');

figure(2);
plot(shp(1:plot_iter,:)); title('First 10 shape coeffs'); grid on; xlim([0 num_iter]);
saveas(2,'fig2.png');
figure(3);
plot(bs(1:plot_iter,:)); title('Blendshape coeffs'); grid on; xlim([0 num_iter]);
saveas(3,'fig3.png');

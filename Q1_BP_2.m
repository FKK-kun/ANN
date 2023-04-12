clear all;
clc;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%读取训练数据 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[a1,a2,a3,a4] = textread('Q1-wind farm data.txt','%s%f%d%f','headerlines',4);
a = [a2 a3 a4];
P1 = a(1:120,1:2);
P2 = a(1:120,3);
P = P1';
T = P2';
T1 = a(121:151,1:2);
T2 = a(121:151,3);
M = T1';
N = T2';
 
%%%%%%%%%%%%%%%%%%%%%特征值归一化 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%[train_data,minp,maxp,train_label,mint,maxt]=premnmx(P,T);
%[test_data,minp,maxp,test_label,mint,maxt]=premnmx(M,N);

% 为了方便反归一化，使用这种方法进行归一化
[train_data, PS_train_data] = mapminmax(P,-1,1);
[train_label, PS_train_label] = mapminmax(T, -1, 1);
[test_data, PS_test_data] = mapminmax(M, -1, 1);
[test_label, PS_test_label] = mapminmax(N, -1, 1);
 
n = 10; %隐藏层单元
num_test_data = size(test_label,2);
%%%%%%%%%%%%%%%%%%%%%%训练%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
net = newff(minmax(train_data),[n,1],{'tansig' 'purelin'},'trainlm');
lr = 0.01;
net.trainParam.epochs = 6000; % 训练网络
net.trainParam.goal = 0.01;
tic
[net,lr] = train(net,train_data,train_label);
toc
Y1 = sim(net,train_data);
Y2 = sim(net,test_data); %训练后的网络进行仿真
e = test_label-Y2;    %误差
Time = 1:31;
%%%%%%%%%%%%%%%%%%%%%%%%%%画图%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure(1)
plot(Time,Y2,Time,test_label,'r-.');
legend('预测结果','目标结果');
title('BP网络数据预处理后预测结果与真实值对比图');
xlabel('时间');
ylabel('预测值');
figure(2)
plot(Time,e);
xlabel('时间');
ylabel('误差');
title('BP网络数据预处理后的误差曲线'); 

% 误差分析
% 进行误差分析之前，需要将数据进行反归一化，这样方便与未进行归一化的结果进行对比
test_label_ori = mapminmax('reverse', test_label, PS_test_label);
Y2_ori = mapminmax('reverse', Y2, PS_test_label);
RMSE = sqrt(sum((test_label_ori - Y2_ori).^2)/num_test_data);  %均方根误差
fprintf('均方根误差：%f', RMSE);
MRE = sum( abs(Y2_ori ./ test_label_ori)) /num_test_data;  %平均相对误差
fprintf('  平均相对误差：%f', MRE);
MD = sum( abs(Y2_ori- test_label_ori)) / num_test_data;  %平均离差
fprintf('  平均离差：%f', MD);
correlation = corrcoef(Y2_ori, test_label_ori);
correlation_coefficient = correlation(1, 2);  %相关系数
fprintf('  相关系数：%f', correlation_coefficient);
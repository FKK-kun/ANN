clear all;
clc;
% 读取数据
[a1,a2,a3,a4] = textread('Q1-wind farm data.txt','%s%f%d%f','headerlines',4);
a = [a2 a3 a4];
P1 = a(1:120,1:2);
P2 = a(1:120,3);
train_data = P1';
train_label = P2';
T1 = a(121:151,1:2);
T2 = a(121:151,3);
test_data = T1';
test_label = T2';
n = 10; %隐藏层单元
num_test_data = size(test_label,2);
 
% 网络初始化
lr = 0.99 * maxlinlr(train_data);
net = newff(minmax(train_data),[n,1],{'tansig' 'purelin'},'trainlm');
net.trainParam.epochs = 6000; % 训练网络
% net.trainParam.goal = 0.001;
tic;
[net,lr] = train(net,train_data,train_label);
toc;
Y1 = sim(net,train_data);
Y2 = sim(net,test_data); %训练后的网络进行仿真
e = test_label-Y2;    %误差
Time = 1:31;


figure(1)
plot(Time,Y2,Time,test_label,'r-.');
legend('预测结果','目标结果');
title('BP网络数据预处理前预测结果与真实值对比图');
xlabel('时间');
ylabel('预测值');
figure(2)
plot(Time,e);
xlabel('时间');
ylabel('误差');
title('BP网络数据预处理前的误差曲线');

% 误差分析
RMSE = sqrt(sum((test_label -Y2).^2)/num_test_data);  %均方根误差
fprintf('均方根误差：%f', RMSE);
MRE = sum( abs(Y2 ./ test_label)) /num_test_data;  %平均相对误差
fprintf('  平均相对误差：%f', MRE);
MD = sum( abs(Y2- test_label)) / num_test_data;  %平均离差
fprintf('  平均离差：%f', MD);
correlation = corrcoef(Y2, test_label);
correlation_coefficient = correlation(1, 2);  %相关系数
fprintf('  相关系数：%f', correlation_coefficient);
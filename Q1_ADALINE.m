clear all;
clc;

% 读取数据
[a1,a2,a3,a4] = textread('Q1-wind farm data.txt','%s%f%d%f','headerlines',4);
data = [a2 a3 a4];
P1 = data(1:120,1:2);
T1 = data(1:120,3);
train_data = P1';
train_label = T1';
P2 = data(121:151,1:2);
T2 = data(121:151,3);
test_data = P2';
test_label = T2';
num_test_data = size(test_label,2);
 
% 网络初始化
[R, Q] = size(train_data);
[S, Q] = size(train_label);
lr = 0.99 * maxlinlr(train_data);
net = newlin(minmax(train_data), S, [0], lr);
net = init(net);

% 训练网络
net.trainParam.epochs = 6000;    % 设定循环次数
tic
net = train(net,train_data,train_label);
toc

%仿真测试
Y1 = sim(net,train_data);
Y2 = sim(net,test_data);
e = test_label - Y2; 
Time = 1:31;
figure(1)
plot(Time,Y2,Time,test_label,'r-.');
legend('预测值','目标值');
title('预测值与真实值对比图');
xlabel('时间');
ylabel('预测值');
figure(2)
plot(Time,e);
xlabel('时间');
ylabel('误差');
title('误差分析');

% 误差分析
RMSE = sqrt(sum((test_label -Y2).^2)/num_test_data);  % 均方根误差
fprintf('均方根误差：%f', RMSE);
MRE = sum( abs(Y2 ./ test_label)) /num_test_data;  % 平均相对误差
fprintf('  平均相对误差：%f', MRE);
MD = sum( abs(Y2- test_label)) / num_test_data;  % 平均离差
fprintf('  平均离差：%f', MD);
correlation = corrcoef(Y2, test_label);
correlation_coefficient = correlation(1, 2);  % 相关系数
fprintf('  相关系数：%f', correlation_coefficient);
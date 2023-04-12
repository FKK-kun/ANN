clear all;
clc;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%加载数据%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[a]=xlsread('Q2-Haberman Survival Data','A2:D307');
[~,idx]=sort(rand(306,1));
b = a(idx(1:200),:);%训练集
c = a(idx(201:306),:);%测试集
b1 = b(:,1:3);
b2 = b(:,4);
train_data = b1';
train_label = b2';
c1 = c(:,1:3);
c2 = c(:,4);
test_data = c1';
test_label = c2';
num_test_data = size(test_label,2);
 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%训练%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
n = 30; %隐藏层单元
net = newff(minmax(train_data),[n,1],{'tansig' 'purelin'},'trainlm');
lr = 2*maxlinlr(train_data);
net.trainParam.epochs = 1000; % 训练网络
net.trainParam.goal = 0.01;
tic;
[net,lr] = train(net,train_data,train_label);
toc;
%Y1 = sim(net,train_data);
Y2 = sim(net,test_data); %训练后的网络进行仿真
e = test_label-Y2;    %误差
 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%准确率%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
test_label2 = [];
for i = 1:106
    if e(i)>0
        test_label2(i) = 2;
    else 
        test_label2(i) = 1;
    end
end
e2 = test_label2-test_label;
 
j = 0;
for i = 1:106
    if e2(i) == 0
     j = j+1;
    end
end
m = j;
rate = m/num_test_data  %准确率
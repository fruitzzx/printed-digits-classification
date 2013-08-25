clear;

load digits.mat

t=cputime;      %record the start time of the program
s=0;            %counter to sum the accuarcy of the running cycles
rts=1;          %Times to train and test the Bagging MLP
k = 5;

disp(['k =' num2str(k)]);
disp('Start...');

for loops=1:rts
    
%cross validate 80% for training and 20% for testing 
[train test] = crossvalind('HoldOut',size(X,1),0.2); 
    
 xtrain = X(train,:);
 xtest  = X(test,:);
 t_train = Y(train,:);
 t_test = Y(test,:);
 
 %reconstruct the t_train output into 4 dimensions
 t_train_plus=zeros(size(t_train,1),4);
 for i=1:size(t_train,1)
            if t_train(i,1) == 1
                t_train_plus(i,1)=1;
                
            elseif t_train(i,1) == 2
                t_train_plus(i,2)= 1;
                
            elseif t_train(i,1) == 3
                t_train_plus(i,3)= 1;
                
            elseif t_train(i,1) == 4
                t_train_plus(i,4)= 1;
            end
 end 

 %using knn and knnfwd to classify the training data	
 net = knn(size(xtrain,2),size(t_train_plus,2),k,xtrain,t_train_plus);
 y = knnfwd(net,xtest);
 
 %translate output of target dataset into one demension (1~4)
 m_t = y';
 [M,I]=max(m_t);
 result = I';
 
 %Evaluation function to get accuracy of network
 s=sum(result==t_test)/size(t_test,1)+s;
  
end

e =cputime -t;
disp('Finshing!');
disp(['Average Accuracy KNN is: ' num2str(s/rts)]);
disp(['Average CPU runningtime of KNN is: ' num2str(e/rts)]); 
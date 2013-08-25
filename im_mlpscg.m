    clear

    load digits.mat

    t=cputime;      %record the start time of the program
    s=0;            %counter to sum the accuarcy of the training cycle
    rts=1;         %running times of the process
   
    %Premeters fo  nwtwork 
    nin = 600;             %Number of inputs
    nhidden = 100;         %Number of hidden units.
    nout = 4;              %Number of outputs.
    alpha = 0.01;	       %Coefficient of weight-decay prior.

    disp(['Number of hidden nodes=' num2str(nhidden)]);
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
    
    
    
    % Create and initialize network weight vector.
    net = mlp(nin, nhidden, nout, 'linear', alpha);
    
    % Set up vector of options for the optimiser.
    options = zeros(size(xtrain,1),1);
    options(1) = 1;			% This provides display of error values.
    options(14) = 100;		% Number of training cycles. 
    
    % Train using scaled conjugate gradients.
    [net, options] = netopt(net, options, xtrain, t_train_plus, 'scg');
   
    % Plot the data, the original function, and the trained network function.
    y = mlpfwd(net, xtest);
    
    
    %translate output of target dataset into one demension (1~4)
     m_t = y';
     [M,I]=max(m_t);
     result = I';
     
     %Evaluation function to get accuracy of network
     s=sum(result==t_test)/size(t_test,1)+s;
     
    end
  
    e =cputime -t;
    
    disp('Finshing!');
    disp(['Average Accuracy of MLP with SCG is: ' num2str(s/rts)]);
    disp(['Average CPU runningtime of MLP with SCG is: ' num2str(e/rts)]); 
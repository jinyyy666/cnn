function average_error = gradCheck(fun, theta0, num_checks, varargin)

 
  sum_error=0;

  n = 100000;
  
  for i=1:num_checks
    T = theta0;
    j = randsample(numel(T), n);
    %T1=T; T1(j) = T1(j)+delta_w;

    [f,g] = fun(T, varargin{:});
    delta_w = 0.001*(2* (g(j) > 0) - 1);
    T0=T; T0(j) = T0(j)-delta_w;
    
    f0 = fun(T0, varargin{:});
    %f1 = fun(T1, varargin{:});

    g_act = (f-f0);
    
    g_cal = g(j)' * delta_w;
    error = 0;
    if(g_act ~= 0)
        error = (abs(g_cal)  - abs(g_act) )/ abs(g_act);
    end

    fprintf('The actual cost change: %f\n', g_act );
    fprintf('The computed cost change: %f\n', g_cal );
    fprintf('The error: %f\n', error);
    %fprintf('% 5d  % 6d % 15g % 15f % 15f % 15f\n', ...
            %i,j,error,g(j),g_est,f);

    sum_error = sum_error + error;
  end

  average_error=sum_error/num_checks;

function [effect] = accumulateEffect(output, input)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Given the input and output spikes, compute the accumulated synaptic
% effect
%
% Input: output: output spikes (1 x end_time)
%        input : input spikes (1 x end_time)
%        
% Output: effect : the accumulated effect
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    tm = 64;
    ts = 8;
    t_ref = 2;
    pre_times = find(input(:) == 1);
    post_times = find(output(:) == 1);
    n_ospikes = sum(output);
    n_ispikes = sum(input);
    
    effect = 0;
    % for each time point when the output fires
    for i = 1:length(post_times)
        t_post = post_times(i);
        if(i == 1)
            last_post = 1;
        else
            last_post = post_times(i-1) + t_ref; 
        end
        lb = max(1, t_post - 4*tm);
        ub = t_post;
        effect_spike_times = pre_times(pre_times >= lb & pre_times < ub) + t_ref; % compensate the effect of refractory period
        
        if(isempty(effect_spike_times))
            continue;
        end
        
        [r, c] = size(effect_spike_times);        
        ss = (t_post - last_post) * ones(r, c);
        tt = t_post * ones(r, c) - effect_spike_times;
        factor = exp(-max(tt - ss, 0)/ts)/(1 - ts/tm);
        all_spike_resp = factor .* (exp(-min(ss, tt)/tm) - exp(-min(ss, tt)/ts));
        all_spike_resp(effect_spike_times > t_post) = 0;
        effect = effect + sum(all_spike_resp);
    end
    if(n_ospikes == 0 && n_ispikes ~= 0)
        effect = 0.1;
    end
end




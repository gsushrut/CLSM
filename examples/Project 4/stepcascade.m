%Checked OK on 20-Apr-2016
%Kills a network by probabilistic Watts' model with a given gamma, epslilon,
%threshold fraction, and number of times.
%Records population and critical times.

matri=conn;

population=zeros(nexp,tmax);        %Population at all times in all cascades

rowdeg=zeros(N,1);                  %Record the initial indegree of each node
for i=1:size(matri)
    rowdeg(i)=sum(matri(i,:))-1;
end

times=zeros(nexp,size(matri,1));    %Death time of each node in each cascade
parfor m=1:nexp                        %For all cascades (m-th cascade)
   matr=matri;                      %Create a duplicate of the network to be killed
   tempx = zeros(1,tmax); 
   for t=1:tmax-1                  %For all time stepsl (t-th time step)
%     population(m,t)=trace(matr);    %Population at that time is the trace of the matrix.
    
    tempx(t) = trace(matr);
    %Population at t is the number of nodes which were inactive at the beginning of the t time.
    %(it is not affected by nodes which activate at t)
    
%    t_tcrit = zeros(1,size(matr,1));
    t_times = zeros(1,size(matr,1));
        for i=1:size(matr,1)        %For each node (i-th node)
 %           if rowdeg(i)~=0         %Non-zero indegree only
 %           if (sum(matr(i,:))-matr(i,i))>(1-fra)*rowdeg(i) %If threshold fraction is not crossed
 %               t_tcrit(i)=t+1;       %this number increases in every step till node gets critical(even activated nodes are declared critical)
 %           end
 %          end
            
        if matr(i,i)~=0             %If a node is inactive
           tmp=rand(1,1);           %Generate random number
           if rowdeg(i)~=0          %Non-zero initial indegree
           if (sum(matr(i,:))-1)<=(1-fra)*rowdeg(i) %If node is critical
               
               zu=rand(1,1);        %Kill node with probability epslilon
               if zu<epsl
                matr(i,i)=0;
                t_times(i)=t;       %If it activates, record this time
               end
           end
           end
           
           if (tmp<gma)           %Irrrespective of condition, kill with probability gamma
                matr(i,i)=0;        %Works for zero initial indegree nodes as well
                t_times(i)=t;       %If it activates, record time
           end
        end                         %End if a node is inactive
        end                         %End for i-th node (go to next node)
        %disp(['m:' int2str(m) ' ss: ' int2str(sum(t_times))]);
%        tcrit(m,:) = tcrit(m,:) + t_tcrit;
        times(m,:) = times(m,:) + t_times;
        for j=1:size(matr,1)        %If a node activated,
           if matr(j,j)==0          %Erase its connections with others
               matr(:,j)=zeros;
           end
        end
    end                             %End t-th time (go to next time step)
    tempx(tmax) = trace(matr);
%     population(m,tmax)=trace(matr); %Define population at last time step
    population(m,:) = tempx;
end                                 %End m-th cascade (go to next cascade)


pavg=ones(1,tmax);                  %Average fraction of nodes inactive at a time t
for i=1:tmax
    pavg(i)=pavg(i)-(mean(population(:,i))/N);
end


pat=zeros(1,tmax);                %Average number of nodes which activate at time t
pat(tmax)=0;
for i=1:tmax-1
    pat(i)=pavg(i+1)-pavg(i);
end
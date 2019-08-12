%This program simulates a cascade on a network, the function g must be
%entered in the 'genfunction' file. The network must be a square matrix variable named
%'conn' with all diagonal entries as 1, and the remaining terms are the
%terms of the adjacency matrix. nexp is the number of experiments to be
%simulated, tmax is the time limit of each experiment. gma is the
%probability that a node activates without interactions.
%This simulation also records the activation times of all nodes, the
%average population of active nodes and the average number of activations
%at all times is also recorded.

N=size(conn,1);
matri=conn;
population=zeros(nexp,tmax);        %Population of inactive nodes at all times in all cascades
rowdeg=zeros(N,1);                  %Record the initial indegree of each node
for i=1:size(matri)
    rowdeg(i)=sum(matri(i,:))-1;
end

times=zeros(nexp,size(matri,1));    %Activation time of each node in each cascade
for m=1:nexp                        %For each cascade (m-th cascade)
   matr=matri;                      %Create a duplicate of the network to be killed
    for t=1:tmax-1                  %For all times
    population(m,t)=trace(matr);    %Population at that time is the trace of the matrix.
    %Population at t is the number of nodes which were inactive at the beginning of the t time.
    
        for i=1:size(matr,1)        %For each node (i-th node)
        if matr(i,i)~=0             %If a node is inactive
           tmp=rand(2,1);           %Generate two random numbers
           if rowdeg(i)~=0          %Non-zero initial indegree
                limt=genfunction((rowdeg(i)-sum(matr(i,:))+1),(rowdeg(i)));
           else limt=0;
           end
           
           if (tmp(1)<limt)        %Activate with probability limt
               matr(i,i)=0;        
               times(m,i)=t;       %If it activates, record time
           end
           
           if (tmp(2)<gma)         %Activate with probability gma
               matr(i,i)=0;        %Works for zero initial indegree nodes as well
               times(m,i)=t;       %If it acctivatess, record time
           end
           
        end                         %End if a node is inactive
        end                         %End for i-th node (go to next node)
        
        for j=1:size(matr,1)        %If a node activates,
           if matr(j,j)==0          %Indicate this to all other nodes connected to it
               matr(:,j)=zeros;
           end
        end
    end
    population(m,tmax)=trace(matr); %End t-th time (go to next time step)
end                                 %End m-th cascade (go to next cascade)
     %Define population at last time step


pavg=ones(1,tmax);                  %Average fraction of nodes active at a time t
for i=1:tmax
    pavg(i)=pavg(i)-(mean(population(:,i))/N);
end


pat=zeros(1,tmax);                %Average number of nodes which activate at time t
pat(tmax)=0;
for i=1:tmax-1
    pat(i)=pavg(i+1)-pavg(i);
end
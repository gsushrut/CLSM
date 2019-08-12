%Gives the indegree distribution of a network
%'conn' must be the network with all diagonal entries as 1 and remaining
%entries must be the entries of the adjacency matrix
matr=conn;
szm=size(matr,1);
rowdeg=zeros(size(matr,1),1);   %We will sum individual rows and store in this column matrix  
parfor i=1:szm
    rowdeg(i)=sum(matr(i,:))-1; %Sum of a i-th row. 1 is deducted due to the diagonal element
end
degdis=[];
parfor i=0:szm            %Count number of times each sum occurs
   k=0;
   for j=1:size(matr,1)
       if i==rowdeg(j)
           k=k+1;
       end
   end
   [degdis]=[degdis k];
end

nordeg=degdis/N;                %Normalized degree distribution
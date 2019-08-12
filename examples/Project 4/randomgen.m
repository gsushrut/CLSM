conn=eye(N);  %This is the connection matrix
for i=1:N
    for j=1:N
        temp=rand;  %Generates a random number between 0 and 1
        if (temp < f)
            conn(i,j)=1;    %Creates an edge with the given probability
        end
    end
end

%------------------------------------------------------------------------%
%----PROJECT - 1--- COMPUTATIONAL LABORATORY IN STATISTICAL MECHANICS----%
%1. GENERATING N UNIFORM RANDOM NUMBERS
%2. TRANSFORMING THE RANDOM VARIABLES ACCORDING TO A GIVEN PDF
%3. ADDING RANDOM NUMBERS AND CENTRAL LIMIT THEOREM
%4. OPEN ENDED: MULTIPLYING/SQUARING RANDOM NUMBERS??



%Number of random numbers
N=1000000;

%Uniform random numbers
r1=rand(N,1);
r8=rand(N,8);
r64=rand(N,64);
r512=rand(N,512);

%p(x)=2x distributed random numbers
q1=r1.^(1/2);
q8=r8.^(1/2);
q64=r64.^(1/2);
q512=r512.^(1/2);

%Sum of 8, 64 and 512 uniform random numbers
t8=sum(r8,2);
t64=sum(r64,2);
t512=sum(r512,2);

%Sum of 8, 64 and 512 p(x)=2x random numbers
s8=sum(q8,2);
s64=sum(q64,2);
s512=sum(q512,2);

%Raw histogram of sums
%rh8=histogram(s8,1000);
%savefig('rhist8.fig');
%rh64=histogram(s64,1000);
%savefig('rhist64.fig');
%rh512=histogram(s512,1000);
%savefig('rhist512.fig');

%Scaled distributions for uniform p(x)
w8=(t8-mean(t8))./(std(t8));
w64=(t64-mean(t64))./(std(t64));
w512=(t512-mean(t512))./(std(t512));

%Scaled distributions for p(x)=2x
y8=(s8-mean(s8))./(std(s8));
y64=(s64-mean(s64))./(std(s64));
y512=(s512-mean(s512))./(std(s512));


%Comparison with Gaussian
avec=-5:0.001:5;
xvec=transpose(avec);
ygaus=normpdf(xvec,0,1);

%Histograms of scaled distributions
hold off
hold on
sh8=histogram(y8,1000,'Normalization','pdf');
th8=histogram(w8,1000,'Normalization','pdf');
ga8=plot(xvec,ygaus,'LineWidth',3);
savefig('shist8.fig');
hold off
hold on
sh64=histogram(y64,1000,'Normalization','pdf');
th64=histogram(w64,1000,'Normalization','pdf');
ga64=plot(xvec,ygaus);
savefig('shist64.fig','LineWidth',3);
hold off
hold on
sh512=histogram(y512,1000,'Normalization','pdf');
th512=histogram(w512,1000,'Normalization','pdf');
ga512=plot(xvec,ygaus);
savefig('shist512.fig','LineWidth',3);
hold off
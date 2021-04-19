% Convert ANN wb matrix to a graph theory V (vertex) matrix
clear all

% How many input neurons?
m= 31;
% How many hidden neurons?
p= 10;
% How many output neurons?
n= 5;
% Initialize V matrix
x = m+p+n;
V = zeros(x,x);

% Load net and wb matrix
load aR8BRnet
load aR8_BR_wb.mat
net = aR8BRnet;
% ^^^ MAKE SURE YOU HAVE THE CORRECT VARIABLES

% Separate out the weights between layers and neuron biases
[b,iw,lw] = separatewb(net,wb);

% Need to fit to graph dimensioning
iwv=transpose(iw{1,1});
lwv=transpose(lw{2,1});

% Add iw (input to hidden) and lw (hidden to output) into network at
% respective location
V(1:m,m+1:m+p)=iwv;
V(m+1:m+p,m+p+1:x)=lwv;

save('V_RM8a.mat', 'V');

% Create matrix that replaces any negative value with a zero value
row = size(V,1);
col = size(V,2);
VRN = zeros;
for r = 1:row
    for c = 1:col
        if V(r,c)<0
            VRN(r,c)=0;
        else
            VRN(r,c) = V(r,c);
        end
    end
end

% Save new matrix
save('V_RM8aRN.mat', 'VRN');



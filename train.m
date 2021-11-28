function [Wx, Wy, R, B, Wx1, Wy1, R1, B10, Wx2, Wy2, R2, B2, Wx3, Wy3, R3, B3] = train(X, Y, param, L)

fprintf('training...\n');

%% set the parameters
nbits = param.nbits;
nbits1 = param.nbits1;
nbits2 = param.nbits2;
nbits3 = param.nbits3;


lambdaX = param.lambdaX;
lambdaY = 1-lambdaX;
alpha = param.alpha;
gamma = param.gamma;
Xbeide = param.Xbeide;
Xmu = param.Xmu;
Ymu = Xmu;

%% get the dimensions
[n, dX] = size(X);
dY = size(Y,2);

%% transpose the matrices
X = X'; Y = Y'; L = L';

%% initialization
V = randn(nbits, n);
Wx = randn(nbits, dX);
Wy = randn(nbits, dY);
R = randn(nbits, nbits);
[U11, ~, ~] = svd(R);
R = U11(:,1:nbits);
T = randn(nbits, nbits1);

V1 = randn(nbits1, n);
Wx1 = randn(nbits1, dX);
Wy1 = randn(nbits1, dY);
R1 = randn(nbits1, nbits);
[U111, ~, ~] = svd(R1);
R1 = U111(:,1:nbits1);
T1 = randn(nbits1, nbits2);


V2 = randn(nbits2, n);
Wx2 = randn(nbits2, dX);
Wy2 = randn(nbits2, dY);
R2 = randn(nbits2, nbits);
[U112, ~, ~] = svd(R2);
R2 = U112(:,1:nbits2);
T2 = randn(nbits2, nbits3);


V3 = randn(nbits3, n);
Wx3 = randn(nbits3, dX);
Wy3 = randn(nbits3, dY);
R3 = randn(nbits3, nbits);
[U113, ~, ~] = svd(R3);
R3 = U113(:,1:nbits3);


B10 = sign(randn(nbits1,n)); B10(B10==0) = -1;
B2 =  sign(randn(nbits2,n)); B2(B2==0) = -1;
B3 =  sign(randn(nbits3,n)); B3(B3==0) = -1;


%% iterative optimization
for iter = 1:param.iter

    % update B
    B = -1*ones(nbits,n);
    B((R*V+Xbeide*T*B10)>=0) = 1;
    % update T
    T = Xbeide*B*B10'/(Xbeide*B10*B10'+0.01*eye(nbits1));
    % update G
    Ux = lambdaX*(X*V')/(lambdaX*(V*V')+gamma*eye(nbits));
    Uy = lambdaY*(Y*V')/(lambdaY*(V*V')+gamma*eye(nbits));
    G = alpha*(L*V')/(alpha*(V*V')+gamma*eye(nbits));

    % update W
    Wx = Xmu*(V*X')/(Xmu*(X*X')+gamma*eye(dX));
    Wy = Ymu*(V*Y')/(Ymu*(Y*Y')+gamma*eye(dY));

    % update V
    V = (lambdaX*(Ux'*Ux)+lambdaY*(Uy'*Uy)+alpha*(G'*G)+(R'*R)+(Xmu+Ymu+gamma)*eye(nbits))\(lambdaX*(Ux'*X)+lambdaY*(Uy'*Y)+Xmu*(Wx*X)+Ymu*(Wy*Y)+alpha*(G'*L)+(R'*B));

    % update R
    [S1, ~, S2] = svd(B*V');
    R = S1*S2';
%%  updata B1.
      % update B
    B10 = -1*ones(nbits1,n);
    B10((R1*V1+Xbeide*T1*B2)>=0) = 1;
    % update T1
    T1 = Xbeide*B10*B2'/(Xbeide*B2*B2'+0.01*eye(nbits2));

    
    % update G
    Ux1 = lambdaX*(X*V1')/(lambdaX*(V1*V1')+gamma*eye(nbits1));
    Uy1 = lambdaY*(Y*V1')/(lambdaY*(V1*V1')+gamma*eye(nbits1));
    G1 = alpha*(L*V1')/(alpha*(V1*V1')+gamma*eye(nbits1));

    % update W
    Wx1 = Xmu*(V1*X')/(Xmu*(X*X')+gamma*eye(dX));
    Wy1 = Ymu*(V1*Y')/(Ymu*(Y*Y')+gamma*eye(dY));

    % update V
    V1 = (lambdaX*(Ux1'*Ux1)+lambdaY*(Uy1'*Uy1)+alpha*(G1'*G1)+(R1'*R1)+(Xmu+Ymu+gamma)*eye(nbits1))\(lambdaX*(Ux1'*X)+lambdaY*(Uy1'*Y)+Xmu*(Wx1*X)+Ymu*(Wy1*Y)+alpha*(G1'*L)+(R1'*B10));

    % update R
    [S11, ~, S21] = svd(B10*V1');
    R1 = S11*S21';


%% UPDATE B2


  % update B
    B2 = -1*ones(nbits2,n);
    B2((R2*V2+Xbeide*T2*B3)>=0) = 1;
    
    T2 = Xbeide*B2*B3'/(Xbeide*B3*B3'+0.01*eye(nbits3));

    % update G
    Ux2 = lambdaX*(X*V2')/(lambdaX*(V2*V2')+gamma*eye(nbits2));
    Uy2 = lambdaY*(Y*V2')/(lambdaY*(V2*V2')+gamma*eye(nbits2));
    G2 = alpha*(L*V2')/(alpha*(V2*V2')+gamma*eye(nbits2));

    % update W
    Wx2 = Xmu*(V2*X')/(Xmu*(X*X')+gamma*eye(dX));
    Wy2 = Ymu*(V2*Y')/(Ymu*(Y*Y')+gamma*eye(dY));

    % update V
    V2 = (lambdaX*(Ux2'*Ux2)+lambdaY*(Uy2'*Uy2)+alpha*(G2'*G2)+(R2'*R2)+(Xmu+Ymu+gamma)*eye(nbits2))\(lambdaX*(Ux2'*X)+lambdaY*(Uy2'*Y)+Xmu*(Wx2*X)+Ymu*(Wy2*Y)+alpha*(G2'*L)+(R2'*B2));

    % update R
    [S12, ~, S22] = svd(B2*V2');
    R2 = S12*S22';
    
    %% UPDATE B3
  % update B
    B3 = -1*ones(nbits3,n);
    B3((R3*V3)>=0) = 1;
    % update T
    T3 = Xbeide*B*B10'/(Xbeide*B10*B10'+0.01*eye(nbits1));
    % update G
    Ux3 = lambdaX*(X*V3')/(lambdaX*(V3*V3')+gamma*eye(nbits3));
    Uy3 = lambdaY*(Y*V3')/(lambdaY*(V3*V3')+gamma*eye(nbits3));
    G3 = alpha*(L*V3')/(alpha*(V3*V3')+gamma*eye(nbits3));

    % update W
    Wx3 = Xmu*(V3*X')/(Xmu*(X*X')+gamma*eye(dX));
    Wy3 = Ymu*(V3*Y')/(Ymu*(Y*Y')+gamma*eye(dY));

    % update V
    V3 = (lambdaX*(Ux3'*Ux3)+lambdaY*(Uy3'*Uy3)+alpha*(G3'*G3)+(R3'*R3)+(Xmu+Ymu+gamma)*eye(nbits3))\(lambdaX*(Ux3'*X)+lambdaY*(Uy3'*Y)+Xmu*(Wx3*X)+Ymu*(Wy3*Y)+alpha*(G3'*L)+(R3'*B3));

    % update R
    [S13, ~, S23] = svd(B3*V3');
    R3 = S13*S23';
    
    
    
end

%clear all

gpuDevice();
profile on
%% GPU configuration
gpuInfo = gpuDevice;
fprintf('GPU Name: %s\n', gpuInfo.Name);
fprintf('Total Memory: %.2f GB\n', gpuInfo.TotalMemory / 1e9);
fprintf('Compute Capability: %.1f\n', gpuInfo.ComputeCapability);
fprintf('Clock Rate: %.2f MHz\n', gpuInfo.ClockRateKHz / 1000);
fprintf('Max Threads Per Block: %d\n', gpuInfo.MaxThreadsPerBlock);

%% Data
rng(1)
P = 100; % Number of rows in the orthogonal matrix
Q = 100; % <= P; Number of columns in the orthogonal matrix
M = 1000; % >= max(P,Q)
B = randn(Q, M); % B is Q x M
[O_true, ~] = qr(randn(P, Q), 0); % O is P x Q column orthogonal
% Procrustes problem: ||A -O*B|| where O is column orthogonal
A = O_true * B; % P x M

%% Parameters
MaxTime = 3600;
MaxRuns = 10;
MaxIter = 1000;
rho = 2;
TolFun1 = 10^(-4);
TolFun2 = 10^(-4);
phi = 10^(-10);
ub = pi;
lb = 0;
sIntitial = 1;
thetaInitial = pi*sIntitial;
DisplayUpdate = 1;
DisplayEvery = 2;
PrintStepSize = 1;
%PrintSolution = 0;

%% Initial point
rng(123)
[O_initial, ~] = qr(randn(P, Q), 0);


%% Transfering all to GPU

AGpu = gpuArray(A); % Transfer A to GPU
BGpu = gpuArray(B); % Transfer B to GPU
O_initial_gpu = gpuArray(O_initial);
theta = pi;
Fun = @(O) Procrustes(AGpu, O, BGpu);
RunSolnArray = nan(MaxRuns,1);
last_toc = 0;
break_now = 0;
fprintf('========================= BOOOM Starts =======================\n')
tic;
for iii = 1:MaxRuns
    theta = thetaInitial;
    if(iii == 1)
        O_updated = O_initial_gpu;
    end
    for i = 1:MaxIter
        if(toc > MaxTime)
            break_now = 1;
            fprintf('=> As requested, MSCOR has been terminated after %.2f seconds :( \n', MaxTime);
            fprintf('\n')
            break;
        end
        O = O_updated;
        InitialValue = Fun(O);
        [pairs_i, pairs_j] = find(triu(ones(Q), 1)); % Upper triangular indices
        num_rotations = length(pairs_i);
        
        rotation_function = @(r_i, r_j) create_rotation_matrix(Q, r_i, r_j, theta);
        
        %%%% Time display %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        toc_now = toc;
        if(DisplayUpdate == 1)
            if(toc_now - last_toc > DisplayEvery)
                if(PrintStepSize == 1)
                    fprintf('=> Executing Run: %d, iter: %d, current obj. fun. value: %d, current log10(step-size/pi): %.2f. \n', iii, i, InitialValue, log10(theta/pi));
                else
                    fprintf('=> Executing Run: %d, iter: %d, current obj. fun. value: %d. \n', iii, i, InitialValue);
                end
                last_toc = toc_now;
            end
        end
        FunValsPosMovesGpu = arrayfun(@(idx) ObjFunRotated(AGpu, O, BGpu, pairs_i(idx), pairs_j(idx), theta), ...
            1:num_rotations);
        FunValsNegMovesGpu = arrayfun(@(idx) ObjFunRotated(AGpu, O, BGpu, pairs_i(idx), pairs_j(idx), -theta), ...
            1:num_rotations);
        
        FunValsPosMoves = gather(FunValsPosMovesGpu);
        FunValsNegMoves = gather(FunValsNegMovesGpu);
        
        [minValuePos, minIndexPos] = min(FunValsPosMoves);
        [minValueNeg, minIndexNeg] = min(FunValsNegMoves);
        
        CurrentValue = InitialValue;
        if(min(minValuePos,minValueNeg) <  InitialValue)
            if(minValuePos < minValueNeg)
                R = create_rotation_matrix(Q, pairs_i(minIndexPos), pairs_j(minIndexPos), theta);
                O_updated = O * R; % column-orthogonality preserved
            else
                R = create_rotation_matrix(Q, pairs_i(minIndexNeg), pairs_j(minIndexNeg), -theta);
                O_updated = O * R; % column-orthogonality preserved
            end
            CurrentValue = Fun(O_updated);
        end
        
        if (i > 1)
            if(abs(CurrentValue - InitialValue) < TolFun1)
                if(theta > phi)
                    theta = theta/rho;
                else
                    break;
                end
            end
        end
        %fprintf('\n')
        %fprintf('=> Objective before update: %.2f, Objective after update: %.2f. \n',InitialValue, CurrentValue);
        
        % [InitialValue, CurrentValue]
        
    end
    RunSolnArray(iii) = CurrentValue;
    
    
    if(iii > 1)
        if(abs(RunSolnArray(iii) - RunSolnArray(iii-1)) < TolFun2)
            break;
        end
    end
    if(break_now == 1)
        break;
    end
end

comp_time = toc;
fprintf('\n')
fprintf('=> Obj. fun. value at BOOOM minima: %d \n',CurrentValue);
fprintf('\n')
fprintf('=> Total time taken: %.4f secs.\n',comp_time);

fprintf('xxxxxxxxxxxxxxxxxxxxxx BOOOM ends xxxxxxxxxxxxxxxxxxxxxxxxxx\n')

%time_gpu = toc;
profile viewer
norm(O_true - O)

% disp(['GPU computation time: ', num2str(time_gpu), ' seconds']);
% disp('L2 norms (first 5 rotations):');
% disp(FunVals(1:5));
function R = create_rotation_matrix(n, r_i, r_j, theta)
    % Create a block rotation matrix for n dimensions
    R = eye(n, 'gpuArray');
    
    % Define the 2D rotation matrix
    cos_t = cos(theta);
    sin_t = sin(theta);
    R([r_i, r_j], [r_i, r_j]) = [cos_t, -sin_t; sin_t, cos_t];
end
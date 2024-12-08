function value = ObjFunRotated(AGpu, O, BGpu, r_i, r_j, theta)
    % Create the rotation matrix for the given pair (i, j)
    R = create_rotation_matrix(size(O, 2), r_i, r_j, theta);
    rotated_O = O * R; % Ensure column-orthogonality is preserved
    value = Procrustes(AGpu, rotated_O, BGpu);
end
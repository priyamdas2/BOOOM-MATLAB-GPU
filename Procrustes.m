function value = Procrustes(AGpu, O, BGpu)
    value = single(norm(AGpu - O*BGpu, 'fro'));
end
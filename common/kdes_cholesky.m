
%from kdes_params, we use .codebook and .emk_params
% preprocess the KDES basis -- look at KDES cksvd_emk_batch() for more detail
function cholesky_result = kdes_cholesky(kdes_params)
    cholesky_result = containers.Map;
    for i=1:length(kdes_params.kdestype) %number of descriptor types in use (grad, lbp, rgb, ...)
        curr_kdestype = kdes_params.kdestype{i};
        basis = kdes_params.codebook{i};
        K = eval_kernel(basis', basis', kdes_params.emk_params.ktype, kdes_params.kparam(i)); %TODO: store in a mat file after precomputing?
        K = K + 1e-6*eye(size(K));
        G = chol(inv(K));
        cholesky_result(curr_kdestype) = G;
    end
end



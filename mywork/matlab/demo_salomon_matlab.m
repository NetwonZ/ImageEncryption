function demo_salomon_matlab()
%DEMO_SALOMON_MATLAB Simple demo for the Salomon CML rd_matrix generator.

    L = 64 * 64;
    N = 28;

    params.mu = 5;
    params.lam = 5;
    params.a = 100;
    params.b = 200;
    params.xi = 1;
    params.eta = 1;

    rng(2026, "twister");
    x0 = rand(1, L);
    z0 = rand();

    rd_matrix = generate_rdseq(x0, z0, params, N, true);

    disp(size(rd_matrix));
    disp(rd_matrix(1, 1:8));
end


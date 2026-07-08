function rd_matrix = generate_rdseq(x0, z0, params, N, is_mod)
%GENERATE_RDSEQ Generate the (N, L) random matrix using the Salomon CML rule.
%
% rd_matrix = generate_rdseq(x0, z0, params, N, is_mod)
% x0      : 1-by-L initial state
% z0      : initial z state (kept for API compatibility; not used)
% params  : struct with fields mu, lam, a, b, xi, eta
% N       : number of iterations
% is_mod  : whether to wrap x into [0, 1)

    if nargin < 5
        is_mod = true;
    end

    validateattributes(x0, {"double"}, {"vector", "real", "finite"});
    validateattributes(z0, {"double"}, {"scalar", "real", "finite"});
    validateattributes(N, {"double"}, {"scalar", "integer", "positive"});

    x0 = double(x0(:).');
    L = numel(x0);

    required = {"mu", "lam", "a", "b", "xi", "eta"};
    for k = 1:numel(required)
        if ~isfield(params, required{k})
            error("Missing required param: %s", required{k});
        end
    end

    mu = double(params.mu);
    a = double(params.a);
    xi = double(params.xi);
    eta = double(params.eta);

    if xi == 0
        eta = L;
    end
    if eta == 0
        xi = L;
    end

    p_idx = mod((1 + xi) * (0:L-1), L) + 1;
    q_idx = mod((eta + xi * eta + 1) * (0:L-1), L) + 1;

    rd_matrix = zeros(N, L);
    x = x0;

    factor = 5 + 3 * mu;
    inner_factor = 15 * pi;

    for t = 1:N
        fx = abs(sin(factor * (1 - (a .* x .* sin(inner_factor .* x .* (1 - x))))));

        fx_left = [fx(end), fx(1:end-1)];
        fx_right = [fx(2:end), fx(1)];
        fx_p = fx(p_idx);
        fx_q = fx(q_idx);

        x_next = 1 - cos(2 * pi * (fx_left + fx + fx_right));
        x_next = x_next + 0.1 * sqrt(fx_p.^2 + fx_q.^2);

        if is_mod
            x_next = mod(x_next, 1.0);
        end

        rd_matrix(t, :) = x_next;
        x = x_next;
    end
end

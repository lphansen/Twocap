#=============================================================================#
#  Economy with TWO CAPITAL STOCKS
#
#  Author: Balint Szoke
#  Date: Sep 2018
#=============================================================================#

using Pkg
using Optim
using Roots
using NPZ
using Distributed
using CSV
using Tables
using ArgParse
using Interpolations

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--gamma"
            help = "gamma"
            arg_type = Float64
            default = 8.0
        "--rho"
            help = "rho"
            arg_type = Float64
            default = 1.00001   
        "--kappa"
            help = "kappa"
            arg_type = Float64
            default = 0.0
        "--zeta"
            help = "zeta"
            arg_type = Float64
            default = 0.5  
        "--Delta"
            help = "Delta"
            arg_type = Float64
            default = 1000.  
        "--symmetric"
            help = "symmetric"
            arg_type = Int
            default = 0
        "--dataname"
            help = "dataname"
            arg_type = String
            default = "output"
        "--llim"
            help = "llim"
            arg_type = Float64
            default = 1.0
        "--srange"
            help = "srange"
            arg_type = Float64
            default = 1.0
        "--lscale"
            help = "lscale"
            arg_type = Float64
            default = 1.0
        "--zscale"
            help = "zscale"
            arg_type = Float64
            default = 1.0
        "--sscale"
            help = "sscale"
            arg_type = Float64
            default = 1.0
        "--foc"
            help = "foc"
            arg_type = Int
            default = 1
        "--clowerlim"
            help = "clowerlim"
            arg_type = Float64
            default = 0.0001
        "--preload"
            help = "preload"
            arg_type = Int
            default = 1
        "--selfpreload"
            help = "selfpreload"
            arg_type = Int
            default = 1
    end
    return parse_args(s)
end

#==============================================================================#
# SPECIFICATION:
#==============================================================================#
@show parsed_args = parse_commandline()
gamma                = parsed_args["gamma"]
rho                  = parsed_args["rho"]
kappa                = parsed_args["kappa"]
zeta                 = parsed_args["zeta"]
Delta                = parsed_args["Delta"]
symmetric            = parsed_args["symmetric"]
dataname             = parsed_args["dataname"]
llim                 = parsed_args["llim"]
srange                = parsed_args["srange"]
lscale                = parsed_args["lscale"]
zscale                = parsed_args["zscale"]
sscale                = parsed_args["sscale"]
foc                   = parsed_args["foc"]
clowerlim             = parsed_args["clowerlim"]
preload               = parsed_args["preload"]
selfpreload           = parsed_args["selfpreload"]

symmetric_returns    = symmetric

include("newsets_utils_stk.jl")

println("=============================================================")
if symmetric_returns == 1
    println(" Economy with two capital stocks: SYMMETRIC RETURNS          ")
    filename = "model_sym_"*string(Delta)*".npz";
elseif symmetric_returns == 0
    println(" Economy with two capital stocks: ASYMMETRIC RETURNS         ")
    filename = "model_asym_"*string(Delta)*".npz";
end

filename_ell = "./output/"*dataname*"/llim_"*string(llim)*"_lscale_"*string(lscale)*"_zscale_"*string(zscale)*"/sscale_"*string(sscale)*"_srange_"*string(srange)*"/kappa_"*string(kappa)*"_zeta_"*string(zeta)*"/gamma_"*string(gamma)*"_rho_"*string(rho)*"/"
isdir(filename_ell) || mkpath(filename_ell)

#==============================================================================#
#  PARAMETERS
#==============================================================================#

# (1) Baseline model
a11 = 0.014
a22 = 0.013
alpha = 0.1

stdscale = sqrt(1.754)
sigma_k1 = stdscale*[.00477,               .0,   .0, .0];
sigma_k2 = stdscale*[.0              , .00477,   .0, .0];
sigma_z =  [.011*sqrt(.5)   , .011*sqrt(.5)   , .025, .0];
# sigma_s =  [.0   , .0   , .0 , .1];
# sigma_s =  [.1*sqrt(.5)   , .1*sqrt(.5)   , .0, .1];
sigma_s =  [.3*sqrt(.5)   , .3*sqrt(.5)   , .0, .3];

eta1 = 0.012790328319261378
eta2 = 0.012790328319261378
if symmetric_returns == 1
    beta1 = 0.01
    beta2 = 0.01
else
    beta1 = 0.0
    beta2 = 0.01
end

delta = .002;

phi1 = 28.0
phi2 = 28.0

# (3) GRID
II, JJ, SS = trunc(Int,1000*lscale+1), trunc(Int,200*zscale+1), trunc(Int,20*sscale+1);
rmax = llim;
rmin = -llim;
zmax = 1.;
zmin = -zmax;
smax = 1. + srange;
smin = 1. - srange;
# smax = 0. + srange;
# smin = 0. - srange;

# (4) Iteration parameters
maxit = 50000;        # maximum number of iterations in the HJB loop
# maxit = 6;        # maximum number of iterations in the HJB loop
crit  = 10e-6;      # criterion HJB loop
# crit  = 10e-8;      # criterion HJB loop
# crit  = 10e-3;      # criterion HJB loop
# Delta = 1000.;      # delta in HJB algorithm


# Initialize model objects -----------------------------------------------------
baseline1 = Baseline(a11, a22, zeta, kappa, sigma_z, sigma_s, beta1, eta1, sigma_k1, delta);
baseline2 = Baseline(a11, a22, zeta, kappa, sigma_z, sigma_s, beta2, eta2, sigma_k2, delta);
technology1 = Technology(alpha, phi1);
technology2 = Technology(alpha, phi2);
model = TwoCapitalEconomy(baseline1, baseline2, technology1, technology2);

grid = Grid_rz(rmin, rmax, II, zmin, zmax, JJ, smin, smax, SS);
params = FinDiffMethod(maxit, crit, Delta);

#==============================================================================#
# WITH ROBUSTNESS
#==============================================================================#

if preload == 1
    if selfpreload == 0
        preload_kappa = kappa
        preload_rho = rho
        if symmetric_returns == 1
            preload_Delta = 300.0
        else
            if kappa == 0.0
                preload_Delta = 100.0
            else
                preload_Delta = 150.0
            end
        end
        # preload_Delta = 150.0
        preload_zeta = 0.5
        preload_llim = llim#1.0
        preload_lscale = 3.0#1.0
        preload_zscale = 1.0
        if kappa == 0.0
            preloadname = "/project/lhansen/twocapkpa/output/"*"twocap_re_calib_kappa_0"*"/Delta_"*string(preload_Delta)*"_llim_"*string(preload_llim)*"_lscale_"*string(preload_lscale)*"_zscale_"*string(preload_zscale)*"/kappa_"*string(preload_kappa)*"_zeta_"*string(preload_zeta)*"/gamma_"*string(gamma)*"_rho_"*string(preload_rho)*"/"
        else
            # preloadname = "/project/lhansen/twocapkpa/output/"*"twocap_re_calib_bal_bet"*"/Delta_"*string(preload_Delta)*"_llim_"*string(preload_llim)*"_lscale_"*string(preload_lscale)*"_zscale_"*string(preload_zscale)*"/kappa_"*string(preload_kappa)*"_zeta_"*string(preload_zeta)*"/gamma_"*string(gamma)*"_rho_"*string(preload_rho)*"/"
            preloadname = "/project/lhansen/twocapkpa/output/"*"twocap_re_calib"*"/Delta_"*string(preload_Delta)*"_llim_"*string(preload_llim)*"_lscale_"*string(preload_lscale)*"_zscale_"*string(preload_zscale)*"/kappa_"*string(preload_kappa)*"_zeta_"*string(preload_zeta)*"/gamma_"*string(gamma)*"_rho_"*string(preload_rho)*"/"
        end
        if symmetric_returns == 1
            preload = npzread(preloadname*"model_sym_HS.npz")
        else
            preload = npzread(preloadname*"model_asym_HS.npz")
        end
        println("preload location : "*preloadname)
        A_x1 = range(-llim,llim,trunc(Int,1000*preload_lscale+1))
        A_x2 = range(-1,1,trunc(Int,200*preload_zscale+1))
        A_rr = range(rmin, stop=rmax, length=II);
        A_zz = range(zmin, stop=zmax, length=JJ);

        println("preload V0 starts")
        preloadV0 = ones(II, JJ, SS)
        itp = interpolate(preload["V"], BSpline(Cubic(Line(OnGrid()))))
        sitp = scale(itp, A_x1, A_x2)
        println("(1,-1): ",sitp(1,-1))
        preloadV0tdm = ones(II, JJ)
        for i = 1:II
            for j = 1:JJ
                preloadV0tdm[i,j] = sitp(A_rr[i], A_zz[j])
            end
        end
        println("(1,-1): ",preloadV0tdm[end,1])
        for i = 1:SS
            preloadV0[:,:,i] = preloadV0tdm
        end
        println("preload V0 ends")
        println("preload d1 starts")
        preloadd1 = ones(II, JJ, SS)
        itp = interpolate(preload["d1"], BSpline(Cubic(Line(OnGrid()))))
        sitp = scale(itp, A_x1, A_x2)
        println("(1,-1): ",sitp(1,-1))
        preloadd1tdm = ones(II, JJ)
        for i = 1:II
            for j = 1:JJ
                preloadd1tdm[i,j] = sitp(A_rr[i], A_zz[j])
            end
        end
        println("(1,-1): ",preloadd1tdm[end,1])
        for i = 1:SS
            preloadd1[:,:,i] = preloadd1tdm
        end
        println("preload d1 ends")
        println("preload cons starts")
        preloadcons = ones(II, JJ, SS)
        itp = interpolate(preload["cons"], BSpline(Cubic(Line(OnGrid()))))
        sitp = scale(itp, A_x1, A_x2)
        println("(1,-1): ",sitp(1,-1))
        preloadconstdm = ones(II, JJ)
        for i = 1:II
            for j = 1:JJ
                preloadconstdm[i,j] = sitp(A_rr[i], A_zz[j])
            end
        end
        println("(1,-1): ",preloadconstdm[end,1])
        for i = 1:SS
            preloadcons[:,:,i] = preloadconstdm
        end
        println("preload cons ends")
        println("preload VrF starts")
        preloadVrF = ones(II, JJ, SS)
        itp = interpolate(preload["Vr_F"], BSpline(Cubic(Line(OnGrid()))))
        sitp = scale(itp, A_x1, A_x2)
        println("(1,-1): ",sitp(1,-1))
        preloadVrFtdm = ones(II, JJ)
        for i = 1:II
            for j = 1:JJ
                preloadVrFtdm[i,j] = sitp(A_rr[i], A_zz[j])
            end
        end
        println("(1,-1): ",preloadVrFtdm[end,1])    
        for i = 1:SS
            preloadVrF[:,:,i] = preloadVrFtdm
        end
        println("preload VrF ends")
        println("preload VrB starts")
        preloadVrB = ones(II, JJ, SS)
        itp = interpolate(preload["Vr_B"], BSpline(Cubic(Line(OnGrid()))))
        sitp = scale(itp, A_x1, A_x2)
        println("(1,-1): ",sitp(1,-1))
        preloadVrBtdm = ones(II, JJ)
        for i = 1:II
            for j = 1:JJ
                preloadVrBtdm[i,j] = sitp(A_rr[i], A_zz[j])
            end
        end
        println("(1,-1): ",preloadVrBtdm[end,1])
        for i = 1:SS
            preloadVrB[:,:,i] = preloadVrBtdm
        end
        println("preload Vr ends")
    elseif selfpreload == 1
        reload_kappa = kappa
        preload_rho = rho#1.00001#rho#1.0#0.67#rho
        preload_Delta = "5.0e-5"#0.0005#1.0#0.1#0.001#0.005#Delta
        preload_zeta = 0.5
        preload_llim = llim
        preload_lscale = lscale
        preload_zscale = zscale
        preload_sscale = sscale
        preload_srange = srange#0.5#0.0001
        # preloadname = "./output/"*"twocap_stv_test_self_init"*"/llim_"*string(preload_llim)*"_lscale_"*string(preload_lscale)*"_zscale_"*string(preload_zscale)*"/sscale_"*string(preload_sscale)*"_srange_"*string(preload_srange)*"/kappa_"*string(reload_kappa)*"_zeta_"*string(preload_zeta)*"/gamma_"*string(gamma)*"_rho_"*string(preload_rho)*"/"
        # preloadname = "./output/"*"twocap_stv_test"*"/llim_"*string(preload_llim)*"_lscale_"*string(preload_lscale)*"_zscale_"*string(preload_zscale)*"/sscale_"*string(preload_sscale)*"_srange_"*string(preload_srange)*"/kappa_"*string(reload_kappa)*"_zeta_"*string(preload_zeta)*"/gamma_"*string(gamma)*"_rho_"*string(preload_rho)*"/"
        preloadname = "/project/lhansen/twocapcol/plot/BFGSiter_100_BFGSfun_1000_large_scale/kappa_10.0_zeta_0.5/gamma_8.0_rho_"*string(preload_rho)*"/layer_width_16_m_layers_4_n_layers_0_activation_tanh/init_point_size_10.0_max_epoch_1000/"
        # preloadname = "./output/"*"twocap_stk_111_all_cov"*"/llim_"*string(preload_llim)*"_lscale_"*string(preload_lscale)*"_zscale_"*string(preload_zscale)*"/sscale_"*string(preload_sscale)*"_srange_"*string(preload_srange)*"/kappa_"*string(reload_kappa)*"_zeta_"*string(preload_zeta)*"/gamma_"*string(gamma)*"_rho_"*string(preload_rho)*"/"
        # preloadname = "./output/"*"twocap_stv_test_large_cov_self_init"*"/llim_"*string(preload_llim)*"_lscale_"*string(preload_lscale)*"_zscale_"*string(preload_zscale)*"/sscale_"*string(preload_sscale)*"_srange_"*string(preload_srange)*"/kappa_"*string(reload_kappa)*"_zeta_"*string(preload_zeta)*"/gamma_"*string(gamma)*"_rho_"*string(preload_rho)*"/"
        if symmetric_returns == 1
            preload = npzread(preloadname*"symmetric_1_action_time_5/FDMpreload.npz")
            # preload = npzread(preloadname*"model_sym_"*string(preload_Delta)*".npz")
        else
            preload = npzread(preloadname*"symmetric_0_action_time_5/FDMpreload.npz")
            # preload = npzread(preloadname*"model_asym_"*string(preload_Delta)*".npz")
        end
        println("preload location : "*preloadname)
        preloadV0 = preload["V0"]
        preloadd1 = preload["d1"]
        preloadcons = preload["cons"]
        preloadVrF = preload["Vr_F"]
        preloadVrB = preload["Vr_B"]
    end
else
    # if rho == 1.0
        preload_kappa = kappa
        preload_rho = rho
        preload_Delta = Delta
        preload_zeta = 0.5
        preload_llim = llim#1.0
        preload_lscale = lscale
        preload_zscale = zscale
        preloadname = "/project/lhansen/twocapkpa/output/"*"twocap_small_grids"*"/Delta_"*string(preload_Delta)*"_llim_"*string(preload_llim)*"_lscale_"*string(preload_lscale)*"_zscale_"*string(preload_zscale)*"/kappa_"*string(preload_kappa)*"_zeta_"*string(preload_zeta)*"/gamma_"*string(gamma)*"_rho_"*string(preload_rho)*"/"
        # preloadname = "/project/lhansen/twocapkpa/output/"*"twocap_re_calib"*"/Delta_"*string(preload_Delta)*"_llim_"*string(preload_llim)*"_lscale_"*string(preload_lscale)*"_zscale_"*string(preload_zscale)*"/kappa_"*string(preload_kappa)*"_zeta_"*string(preload_zeta)*"/gamma_"*string(gamma)*"_rho_"*string(preload_rho)*"/"
        preload = npzread(preloadname*filename)
        preloadV0 = ones(II, JJ, SS)
        for i = 1:SS
            preloadV0[:,:,i] = preload["V"]
        end
        preloadd1 = ones(II, JJ, SS)
        for i = 1:SS
            preloadd1[:,:,i] = preload["d1"]
        end
        preloadcons = ones(II, JJ, SS)
        for i = 1:SS
            preloadcons[:,:,i] = preload["cons"]
        end
        preloadVrF = ones(II, JJ, SS)
        for i = 1:SS
            preloadVrF[:,:,i] = preload["Vr_F"]
        end
        preloadVrB = ones(II, JJ, SS)
        for i = 1:SS
            preloadVrB[:,:,i] = preload["Vr_B"]
        end
end

println(" (3) Compute value function WITH ROBUSTNESS")
times = @elapsed begin
A, V, val, d1_F, d2_F, d1_B, d2_B, h1_F, h2_F, hz_F, hs_F, h1_B, h2_B, hz_B, hs_B,
           mu_1_F, mu_1_B, mu_r_F, mu_r_B, mu_z, mu_s, V0, Vr, Vz, Vs, Vr_F, Vr_B, Vz_B, Vz_F, Vs_B, Vs_F, cF, cB, rrr, zzz, sss, pii, dr, dz, ds =
        value_function_twocapitals(gamma, rho, model, grid, params, preloadV0, preloadd1, preloadcons, preloadVrF, preloadVrB, foc, clowerlim, symmetric_returns);
println("=============================================================")
end
println("Convegence time (minutes): ", times/60)
g = stationary_distribution(A, grid)

# Define Policies object
policies  = PolicyFunctions(d1_F, d2_F, d1_B, d2_B,
                            -h1_F, -h2_F, -hz_F, -hs_F,
                            -h1_B, -h2_B, -hz_B, -hs_B);

# Construct drift terms under the baseline
mu_1 = (mu_1_F + mu_1_B)/2.;
mu_r = (mu_r_F + mu_r_B)/2.;
# ... under the worst-case model
h1_dist = (policies.h1_F + policies.h1_B)/2.;
h2_dist = (policies.h2_F + policies.h2_B)/2.;
hz_dist = (policies.hz_F + policies.hz_B)/2.;
hs_dist = (policies.hs_F + policies.hs_B)/2.;

######
d1 = (policies.d1_F + policies.d1_B)/2;
d2 = (policies.d2_F + policies.d2_B)/2;
h1, h2, hz, hs= -h1_dist, -h2_dist, -hz_dist, -hs_dist;

r = range(rmin, stop=rmax, length=II);    # capital ratio vector
rr = r * ones(1, JJ);
rrr = ones(II, JJ, SS)
for i = 1:SS
    rrr[:,:,i] = rr
end
pii = rrr;
IJS = II*JJ*SS;
k1a = zeros(II,JJ,SS)
k2a = zeros(II,JJ,SS)
for i=1:IJS
    p = pii[i];
    k1a[i] = (1-zeta + zeta*exp.(p*(1-kappa))).^(1/(kappa-1));
    k2a[i] = ((1-zeta)*exp.(p*((kappa-1))) + zeta).^(1/(kappa-1));
end
d1k = d1.*k1a
d2k = d2.*k2a
c = alpha*ones(II,JJ,SS) - d1k - d2k

results = Dict("delta" => delta,
# Two capital stocks
"eta1" => eta1, "eta2" => eta2, "a11"=> a11, "a22"=> a22, 
"beta1" => beta1, "beta2" => beta2,
"k1a" => k1a, "k2a"=> k2a,
"d1k" => d1k, "d2k"=> d2k,
"sigma_k1" => sigma_k1, "sigma_k2" => sigma_k2,
"sigma_z" =>  sigma_z, "sigma_s" =>  sigma_s, "alpha" => alpha, "kappa" => kappa, "zeta" => zeta, "phi1" => phi1, "phi2" => phi2,
"I" => II, "J" => JJ, "S" => SS,
"rmax" => rmax, "rmin" => rmin, "zmax" => zmax, "zmin" => zmin,
"rrr" => rrr, "zzz" => zzz, "sss" => sss, "pii" => pii, "dr" => dr, "dz" => dz,
"maxit" => maxit, "crit" => crit, "Delta" => Delta,
"g" => g,
"cons" => c,
# Robust control under baseline
"V0" => V0, "V" => V, "Vr" => Vr, "Vz" => Vz, "Vs" => Vs, "Vr_F" => Vr_F, "Vr_B" => Vr_B,  "Vz_F" => Vz_F, "Vz_B" => Vz_B, "Vs_F" => Vs_F, "Vs_B" => Vs_B, "val" => val, "gamma" => gamma, "rho" => rho,
"d1" => d1, "d2" => d2, "d1_F" => d1_F, "d2_F" => d2_F, "d1_B" => d1_B, "d2_B" => d2_B, "cF" => cF, "cB" => cB,
"h1_F" => h1_F, "h2_F" => h2_F, "hz_F" => hz_F, "hs_F" => hs_F, "h1_B" => h1_B, "h2_B" => h2_B, "hz_B" => hz_B, "hs_B" => hs_B, 
"mu_1_F" => mu_1_F, "mu_1_B" => mu_1_B, "mu_r_F" => mu_r_F, "mu_r_B" => mu_r_B, 
"h1" => h1, "h2" => h2, "hz" => hz, "hs" => hs, "foc" => foc, "clowerlim" => clowerlim,  "zscale" => zscale, "lscale" => lscale, "sscale" => sscale, "llim" => llim,"srange" => srange,
"times" => times,
"mu_1" => mu_1, "mu_r" => mu_r, "mu_z" => mu_z, "mu_s" => mu_s)

npzwrite(filename_ell*filename, results)

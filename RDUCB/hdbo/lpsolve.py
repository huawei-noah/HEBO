import sys
import json

from lpsolve55 import *

from lpsolve_config import *

### hyper-values
args = {}

if(len(sys.argv) > 1):
    args = json.loads(sys.argv[1])
else:
    args = {"x": [[0.7392635793983017, 0.039187792254320675, 0.2828069625764096, 0.1201965612131689, 0.29614019752214493, 0.11872771895424405, 0.317983179393976, 0.41426299451466997, 0.06414749634878436, 0.6924721193700198, 0.5666014542065752, 0.2653894909394454, 0.5232480534666997, 0.09394051075844168, 0.5759464955561793, 0.9292961975762141, 0.31856895245132366, 0.6674103799636817, 0.13179786240439217, 0.7163272041185655, 0.2894060929472011, 0.18319136200711683, 0.5865129348100832, 0.020107546187493552, 0.8289400292173631, 0.004695476192547066, 0.6778165367962301, 0.27000797319216485, 0.7351940221225949, 0.9621885451174382, 0.24875314351995803, 0.5761573344178369, 0.592041931271839, 0.5722519057908734, 0.2230816326406183, 0.952749011516985, 0.44712537861762736, 0.8464086724711278, 0.6994792753175043, 0.29743695085513366, 0.8137978197024772, 0.39650574084698464, 0.8811031971111616, 0.5812728726358587, 0.8817353618548528, 0.6925315900777659, 0.7252542798196405, 0.5013243819267023, 0.9560836347232239, 0.6439901992296374, 0.4238550485581797, 0.6063932141279244, 0.019193198309333526, 0.30157481667454933, 0.660173537492685, 0.29007760721044407, 0.6180154289988415, 0.42876870094576613, 0.13547406422245023, 0.29828232595603077, 0.5699649107012649, 0.5908727612481732, 0.5743252488495788, 0.6532008198571336, 0.6521032700016889, 0.43141843543397396, 0.896546595851063, 0.36756187004789653, 0.4358649252656268, 0.8919233550156721, 0.8061939890460857, 0.7038885835403663, 0.10022688731230112, 0.9194826137446735]], "mps_path": "~/mps/markshare_5_0.mps", "infinite": 1e+30, "time_limit": 1}
    
def getCombinationValue(param_prefix, args_x):
    return_val = 0
    
    for param in parameters:
        if param_prefix in param:
            return_val += getSingleValue(param, args_x)
    return return_val


def getSingleValue(param_prefix, args_x):
    index = parameters.index(param_prefix)
    
    target = args_x[index]
    _param_map = key_map[index]
    
    if target in _param_map:
        # direct lookup
        return _param_map[target]
    else:
        # very close value, retrive the closest value
        min_match = min(_param_map.keys(), key=lambda k: abs(k-target))
        max_match = max(_param_map.keys(), key=lambda k: abs(k-target))
        
        if abs(min_match - target) > abs(max_match - target):
            return _param_map[max_match]
        else:
            return _param_map[min_match]

# Return the lpsolve('get_objective', lp)
def run_lp_solve(args_x, mps_path):

    #print(args_x, mps_path)
    lp = lpsolve('read_MPS', mps_path)
    lpsolve('set_timeout', lp, args["time_limit"])
    lpsolve('set_verbose', lp, NEUTRAL)
    # lpsolve('set_verbose', lp, FULL)
    lpsolve('set_infinite', lp, args["infinite"])
    
    #=====================================================================================================================

    lpsolve('set_anti_degen', lp, getCombinationValue("antidegen", args_x))
    lpsolve('set_basiscrash', lp, getSingleValue("basis_crash", args_x))
    lpsolve('set_bb_depthlimit', lp, getSingleValue("bb_depth_absolute", args_x) * getSingleValue("bb_depthlimit", args_x))
    lpsolve('set_bb_rule', lp, getSingleValue("bb_rule_1", args_x) + getCombinationValue("bb_rule_2", args_x))
    lpsolve('set_epsb', lp, getSingleValue("eps_b", args_x))
    lpsolve('set_epsd', lp, getSingleValue("eps_d", args_x))
    lpsolve('set_epsel', lp, getSingleValue("eps_el", args_x))
    lpsolve('set_epsperturb', lp, getSingleValue("eps_perturb", args_x))
    lpsolve('set_epspivot', lp, getSingleValue("eps_pivot", args_x))
    lpsolve('set_improve', lp, getCombinationValue("improve", args_x))
    lpsolve('set_maxpivot', lp, getSingleValue("max_pivot", args_x))
    lpsolve('set_mip_gap', lp, getSingleValue("mip_gap_absolute", args_x) , getSingleValue("mip_gap", args_x))
    lpsolve('set_presolve', lp, getCombinationValue("presolve", args_x))
    lpsolve('set_pivoting', lp, getSingleValue("pivoting_1", args_x) + getSingleValue("pivoting_2", args_x))
    lpsolve('set_scalelimit', lp, getSingleValue("scale_limit", args_x))
    lpsolve('set_scaling', lp, getSingleValue("scaling_1", args_x) + getCombinationValue("scaling_2", args_x))
    lpsolve('set_simplextype', lp, getSingleValue("simplex_type", args_x))

    ### - These Parameters that are reliant on the dimension of the problem

    #lpsolve('set_basis', lp, basis)
    #lpsolve('set_var_branch', lp, var_branch)
    #lpsolve('set_var_weights', lp, var_weights)

    ### --- end of hyperparameters --- ###
    #=====================================================================================================================
    
    # http://lpsolve.sourceforge.net/5.0/lp_solve.htm
    lpsolve('solve', lp)
    y = lpsolve('get_objective', lp)
    return y


#sys.stdout.write(str(args["x"]))
sys.stdout.write("RETURN_OBJECTIVE_VALUE:({})".format(str(run_lp_solve(args["x"][0], args["mps_path"]))))

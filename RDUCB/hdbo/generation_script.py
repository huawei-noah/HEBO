#!/usr/bin/env python
# coding: utf-8

# In[141]:


import numpy as np


# In[142]:


### Global list of LPSolveParam objects

pList = []


# In[143]:


### values for sub param type ###

CATEGORIAL_COMBIN = 0
CATEGORIAL_SINGLE_VAL = 1
BOOL_SINGLE_VAL = 2
LIN_RANGE_INT = 3
LIN_RANGE_FLOAT = 4
LOG_RANGE_INT = 5
LOG_RANGE_FLOAT= 6


# In[144]:


class LPSolveParam:
    def __init__(self, paramName, subParams):
        self.name = paramName
        self.subParamList = subParams
        
    def resolveLPparams(self, formatString):
        paramTuple = ()
        
        for subParam in self.subParamList:
            paramTuple = paramTuple + (subParam.resolveLPSetFunc(),)
            
        return formatString.format(*paramTuple)
    
    def generateKeyValues(self):
        keyValuesTuple = ()
        
        for subParam in self.subParamList:
            keyValuesTuple = joiner(keyValuesTuple, subParam.generateKeyValues(prefix_name=(len(self.subParamList) > 1)))
        
        return keyValuesTuple
    

class SubParam:
    def __init__(self, subParamName, subParamType, subParamValues):
        self.name = subParamName
        self.type = subParamType
        self.values = subParamValues
        
    def resolveLPSetFunc(self):
        if self.type == CATEGORIAL_COMBIN:
            return "getCombinationValue(\"{}\", args_x)".format(self.name)
        else:
            return "getSingleValue(\"{}\", args_x)".format(self.name)
        
    def generateKeyValues(self, **kwargs):
        prefixName = kwargs.get('prefix_name', True)
        rounded = kwargs.get('rounded', -1)
        
        if self.type == CATEGORIAL_COMBIN:
            if prefixName:
                return decompressBoolean(self.values, prefix=self.name+"_")
            else:
                return decompressBoolean(self.values)
        elif self.type == CATEGORIAL_SINGLE_VAL:
            return decompressList(self.values, self.name)
        elif self.type == BOOL_SINGLE_VAL:
            return singleBoolean(self.values, self.name)
        elif self.type == LIN_RANGE_INT:
            return linGen(self.values, self.name, rounded=0, integer=True)
        elif self.type == LIN_RANGE_FLOAT:
            if rounded <= -1:
                return linGen(self.values, self.name)
            else:
                return linGen(self.values, self.name, rounded=rounded)
        elif self.type == LOG_RANGE_INT:
            return geoGen(self.values, self.name, rounded=0, integer=True)
        elif self.type == LOG_RANGE_FLOAT:
            if rounded <= -1:
                return geoGen(self.values, self.name)
            else:
                return geoGen(self.values, self.name, rounded=rounded)
        else:
            return ()


# In[149]:


def decompressBoolean(keyMap,**kwargs):
    parameters = []
    booleanMap = []
    hyperValues = []
    typeList = []
    prefix = kwargs.get('prefix', "")
    
    for key in keyMap:
        hyperValues.append(np.array([0.0, 1.0]))
        parameters.append(prefix + key.lower())
        booleanMap.append({0.0:0.0, 1.0:keyMap[key]})
        typeList.append("categorial")
        
        
    return (hyperValues, booleanMap, parameters, typeList)
    
def decompressList(keyMap,prefix):
    parameters = []
    valueMap = [{}]
    hyperValues = []
    typeList = []
    
    mapValues = list(keyMap.values())
    mapValues.sort()
    
    hyperValues.append(np.around(np.linspace(0.0,1.0,len(mapValues)),decimals=5))
    parameters.append(prefix)
    typeList.append("categorial")
    
    for i in range(len(mapValues)):
        valueMap[0][hyperValues[0][i]] = mapValues[i]
    
    return (hyperValues, valueMap, parameters, typeList)

def anydup(thelist):
    seen = set()
    for x in thelist:
        if x in seen: return True
        seen.add(x)
    return False

def linRange(lower, higher, num, prefix, **kwargs):
    parameters = []
    valueMap = [{}]
    hyperValues = []
    typeList = []
    rounded = kwargs.get('rounded', -1)
    integer = kwargs.get('integer', False)
    
    generatedValues = np.linspace(lower, higher, num)
    
    if(rounded >= 0):
        generatedValues = np.around(generatedValues,decimals=round(rounded))
    
    if(integer):
        generatedValues = generatedValues.astype(int)
    
    hyperValues.append(np.around(np.linspace(0.0,1.0,num),decimals=5))
    parameters.append(prefix)
    typeList.append("range")
    
    for i in range(len(generatedValues)):
        valueMap[0][hyperValues[0][i]] = generatedValues[i]
        
    return (hyperValues, valueMap, parameters, typeList)
        
def geoRange(lower, higher, num, prefix, **kwargs):
    parameters = []
    valueMap = [{}]
    hyperValues = []
    typeList = []
    rounded = kwargs.get('rounded', -1)
    integer = kwargs.get('integer', False)
    
    generatedValues = np.geomspace(lower, higher, num)
    
    if(rounded >= 0):
        generatedValues = np.around(generatedValues,decimals=round(rounded))
    
    if(integer):
        generatedValues = generatedValues.astype(int)
    
    hyperValues.append(np.around(np.linspace(0.0,1.0,num),decimals=5))
    parameters.append(prefix)
    typeList.append("range")
    
    for i in range(len(generatedValues)):
        valueMap[0][hyperValues[0][i]] = generatedValues[i]
        
    return (hyperValues, valueMap, parameters, typeList)

def linGen(args, prefix,  **kwargs):
    rounded = kwargs.get('rounded', -1)
    integer = kwargs.get('integer', False)
    
    return linRange(args["lower"], args["higher"], args["num"], prefix, rounded=rounded, integer=integer)

def geoGen(args, prefix,  **kwargs):
    rounded = kwargs.get('rounded', -1)
    integer = kwargs.get('integer', False)
    
    return geoRange(args["lower"], args["higher"], args["num"], prefix, rounded=rounded, integer=integer)

def singleBoolean(args, prefix):
    parameters = [prefix]
    keyMap = [{0.0: args["value_if_false"], 1.0: args["value_if_true"]}]
    hyperValues = [np.array([0.0, 1.0])]
    typeList = ["value"]
    
    return (hyperValues, keyMap, parameters, typeList)

def joiner(*argv):
    parameters = []
    keyMap = []
    hyperValues = []
    typeList = []
    
    for arg in argv:
        if(len(arg) != 0):
            hyperValues += arg[0]
            keyMap += arg[1]
            parameters += arg[2]
            typeList += arg[3]
    
    return (hyperValues, keyMap, parameters, typeList)

def genLPSolveParam(paramName,*argv):
    subParamList = []
    
    for arg in argv:
        subParamList.append(SubParam(arg[0],arg[1],arg[2]))
    
    return LPSolveParam(paramName, subParamList)

def addToPList(paramName, *argv):
    pList.append(genLPSolveParam(paramName, *argv))

def genLPSolveParamSet(paramList):
    content = ""
    header = "#=====================================================================================================================\n### --- set hyperparameters here --- ###\n"
    footer = "\n\n### --- end of hyperparameters --- ###\n#====================================================================================================================="
    
    stcParamSetFormatList = {
        "anti_degen": "{}",
        "basiscrash": "{}",
        "bb_depthlimit": "{} * {}",
        "bb_rule": "{} + {}",
        "improve": "{}",
        "BFP": "",
        "maxpivot": "{}",
        "presolve": "{}",
        "pivoting": "{} + {}",
        "epsb": "{}",
        "epsd": "{}",
        "epsel": "{}",
        "epsint": "{}",
        "epsperturb": "{}",
        "epspivot": "{}",
        "infinite": "{}",
        "mip_gap": "{} , {}",
        "scaling": "{} + {}",
        "scalelimit": "{}",
        "simplextype": "{}"
    }
    
    dynParamSetFormatList = {
        "basis": "",
        "var_branch": "",
        "var_weights": ""
    }
    
    for param in paramList:
        content = content + "\n    lpsolve('set_{}', lp, {})".format(param.name, param.resolveLPparams(stcParamSetFormatList[param.name]))
    
    content = content + "\n\n    ### - These Parameters that are reliant on the dimension of the problem\n"
    
    for key in dynParamSetFormatList:
        content = content + "\n    #lpsolve('set_{}', lp, {})".format(key, key)
    
    return header + content + footer

def genParamKeyValues(paramList):
    fullStack = ()
    
    for param in paramList:
        fullStack = joiner(fullStack, param.generateKeyValues())
    
    return fullStack

def genFaux(hyperValues):
    return [np.random.choice(arrayOfValues,replace=False) for arrayOfValues in hyperValues]


# In[150]:


### --- Values for generation of sub-parameters --- ###

antidegen = {
    "ANTIDEGEN_NONE": 0,	#No anti-degeneracy handling
    "ANTIDEGEN_FIXEDVARS": 1,	#Check if there are equality slacks in the basis and try to drive them out in order to reduce chance of degeneracy in Phase 1
    "ANTIDEGEN_COLUMNCHECK": 2,
    "ANTIDEGEN_STALLING": 4,
    "ANTIDEGEN_NUMFAILURE": 8,
    "ANTIDEGEN_LOSTFEAS": 16,
    "ANTIDEGEN_INFEASIBLE": 32,
    "ANTIDEGEN_DYNAMIC": 64,
    "ANTIDEGEN_DURINGBB": 128,
    "ANTIDEGEN_RHSPERTURB": 256,	#Perturbation of the working RHS at refactorization
    "ANTIDEGEN_BOUNDFLIP": 512	#Limit bound flips that can sometimes contribute to degeneracy in some models
}

basiscrash = {
    "CRASH_NONE": 0,	#No basis crash
    "CRASH_MOSTFEASIBLE": 2,	#Most feasible basis
    "CRASH_LE""ASTDEGENERATE": 3	#Construct a basis that is in some measure the "least degenerate"
}

bb_depth_absolute = {
    "value_if_true": 1,
    "value_if_false": -1
}

bb_depth_limit = {
    "lower":0,
    "higher":50,
    "num":51
}

bb_rule_1 = {
    "NODE_FIRSTSELECT": 0,	#Select lowest indexed non-integer column
    "NODE_GAPSELECT": 1,	#Selection based on distance from the current bounds
    "NODE_RANGESELECT": 2,	#Selection based on the largest current bound
    "NODE_FRACTIONSELECT": 3,	#Selection based on largest fractional value
    "NODE_PSEUDOCOSTSELECT": 4,	#Simple, unweighted pseudo-cost of a variable
    "NODE_PSEUDONONINTSELECT": 5,	#This is an extended pseudo-costing strategy based on minimizing the number of integer infeasibilities
    "NODE_PSEUDORATIOSELECT": 6,	#This is an extended pseudo-costing strategy based on maximizing the normal pseudo-cost divided by the number of infeasibilities. Effectively, it is similar to (the reciprocal of) a cost/benefit ratio
    "NODE_USERSELECT": 7
}
    
#One of these values may be or-ed with one or more of the following values:

# "NODE_RANDOMIZEMODE": 256,	#Adds a randomization factor to the score for any node candicate
bb_rule_2 = {
    "NODE_WEIGHTREVERSEMODE": 8,	#Select by criterion minimum (worst), rather than criterion maximum (best)
    "NODE_BRANCHREVERSEMODE": 16,	#In case when get_bb_floorfirst is BRANCH_AUTOMATIC, select the oposite direction (lower/upper branch) that BRANCH_AUTOMATIC had chosen.
    "NODE_GREEDYMODE": 32,	 
    "NODE_PSEUDOCOSTMODE": 64,	#Toggles between weighting based on pseudocost or objective function value
    "NODE_DEPTHFIRSTMODE": 128,	#Select the node that has already been selected before the most number of times
    "NODE_GUBMODE": 512,	#Enables GUB mode. Still in development and should not be used at this time.
    "NODE_DYNAMICMODE": 1024,	#When NODE_DEPTHFIRSTMODE is selected, switch off this mode when a first solution is found.
    "NODE_RESTARTMODE": 2048,	#Enables regular restarts of pseudocost value calculations
    "NODE_BREADTHFIRSTMODE": 4096,	#Select the node that has been selected before the fewest number of times or not at all
    "NODE_AUTOORDER": 8192,	#Create an "optimal" B&B variable ordering. Can speed up B&B algorithm.
    "NODE_RCOSTFIXING": 16384,	#Do bound tightening during B&B based of reduced cost information
    "NODE_STRONGINIT": 32768	#Initialize pseudo-costs by strong branching

}

eps_b = {
    "lower":1e-15,
    "higher":1e-5,
    "num":100
}

eps_d = {
    "lower":1e-13,
    "higher":1e-4,
    "num":100
}

eps_el = {
    "lower":1e-17,
    "higher":1e-7,
    "num":100
}

eps_int = {
    "lower":1e-12,
    "higher":1e-2,
    "num":100
}

eps_perturb = {
    "lower":1e-10,
    "higher":1e0,
    "num":100
}

eps_pivot = {
    "lower":2e-12,
    "higher":2e-2,
    "num":100
}

improve = {
    "IMPROVE_NONE": 0,	#improve none
    "IMPROVE_SOLUTION": 1,	#Running accuracy measurement of solved equations based on Bx=r (primal simplex), remedy is refactorization.
    "IMPROVE_DUALFEAS": 2,	#Improve initial dual feasibility by bound flips (highly recommended, and default)
    "IMPROVE_THETAGAP": 4,	#Low-cost accuracy monitoring in the dual, remedy is refactorization
    "IMPROVE_BBSIMPLEX": 8	#By default there is a check for primal/dual feasibility at optimum only for the relaxed problem, this also activates the test at the node level
}

infinite = {
    "lower":1e15,
    "higher":1e45,
    "num":150
}

max_pivot = {
    "lower": 1,
    "higher": 500,
    "num": 250
}

mip_gap_absolute = {
    "value_if_true": True,
    "value_if_false": False
}

mip_gap = {
    "lower":1e-16,
    "higher":1e-6,
    "num":100
}

presolve = {
    "PRESOLVE_NONE": 0,	#No presolve at all
    "PRESOLVE_ROWS": 1,	#Presolve rows
    "PRESOLVE_COLS": 2,	#Presolve columns
    "PRESOLVE_LINDEP": 4,	#Eliminate linearly dependent rows
    "PRESOLVE_SOS": 32,	#Convert constraints to SOSes (only SOS1 handled)
    "PRESOLVE_REDUCEMIP": 64,	#If the ph"ase 1 solution process finds that a constraint is redundant then this constraint is deleted. This is no longer active since it is very rare that this is effective, and also that it adds code complications and delayed presolve effects that are not captured properly.
    "PRESOLVE_KNAPSACK": 128,	#Simplification of knapsack-type constraints through addition of an extra variable, which also helps bound the OF
    "PRESOLVE_ELIMEQ2": 256,	#Direct substitution of one variable in 2-element equality constraints; this requires changes to the constraint matrix. Elimeq2 simply eliminates a variable by substitution when you have 2-element equality constraints. This can sometimes cause fill-in of the constraint matrix, and also be a source of rounding errors which can lead to problems in the simplex.
    "PRESOLVE_IMPLIEDFREE": 512,	#Identify implied free variables (releasing their explicit bounds)
    "PRESOLVE_REDUCEGCD": 1024,	#Reduce (tighten) coefficients in integer models based on GCD argument. Reduce GCD is for mixed integer programs where it is possible to adjust the constraint coefficies due to integrality. This can cause the dual objective ("lower bound") to increase and may make it easier to prove optimality.
    "PRESOLVE_PROBEFIX": 2048,	#Attempt to fix binary variables at one of their bounds
    "PRESOLVE_PROBEREDUCE": 4096,	#Attempt to reduce coefficients in binary models
    "PRESOLVE_ROWDOMINATE": 8192,	#Idenfify and delete qualifying constraints that are dominated by others, also fixes variables at a bound
    "PRESOLVE_COLDOMINATE": 16384,	#Deletes variables (mainly binary), that are dominated by others (only one can be non-zero)
    "PRESOLVE_MERGEROWS": 32768,	#Merges neighboring >= or <= constraints when the vectors are otherwise relatively identical into a single ranged constraint
    "PRESOLVE_IMPLIEDSLK": 65536,	#Converts qualifying equalities to inequalities by converting a column singleton variable to slack. The routine also detects implicit duplicate slacks from inequality constraints, fixes and removes the redundant variable. This latter removal also tends to reduce the risk of degeneracy. The combined function of this option can have a dramatic simplifying effect on some models. Implied slacks is when, for example, there is a column singleton (with zero OF) in an equality constraint. In this case, the column can be deleted and the constraint converted to a LE constraint.
    "PRESOLVE_COLFIXDUAL": 131072,	#Variable fixing and removal based on considering signs of the associated dual constraint. Dual fixing is when the (primal) variable can be fixed due to the implied value of the dual being infinite.
    "PRESOLVE_BOUNDS": 262144,	#Does bound tightening based on full-row constraint information. This can assist in tightening the OF bound, eliminate variables and constraints. At the end of presolve, it is checked if any variables can be deemed free, thereby reducing any chance that degeneracy is introduced via this presolve option.
    "PRESOLVE_DUALS": 524288,	#Calculate duals
    "PRESOLVE_SENSDUALS": 1048576	#Calculate sensitivity if there are integer variables
}

pivoting_1 = {
    "PRICER_FIRSTINDEX": 0,	#Select first
    "PRICER_DANTZIG": 1,	#Select according to Dantzig
    "PRICER_DEVEX": 2,	#Devex pricing from Paula Harris
    "PRICER_STEEPESTEDGE": 3	#Steepest Edge
}

#Some of these values can be combined with any (ORed) of the following modes:
#     "PRICE_RANDOMIZE": 128,	#Adds a small randomization effect to the selected pricer
# BUG, keep crashing
# "PRICE_PARTIAL": 16,	#Enable partial pricing
pivoting_2 = {
    "PRICE_PRIMALFALLBACK": 4,	#In case of Steepest Edge, fall back to DEVEX in primal
    "PRICE_MULTIPLE": 8,	#Preliminary implementation of the multiple pricing scheme. This means that attractive candidate entering columns from one iteration may be used in the subsequent iteration, avoiding full updating of reduced costs.  In the current implementation, lp_solve only reuses the 2nd best entering column alternative
    "PRICE_ADAPTIVE": 32,	#Temporarily use alternative strategy if cycling is detected
    "PRICE_AUTOPARTIAL": 512,	#Indicates automatic detection of segmented/staged/blocked models. It refers to partial pricing rather than full pricing. With full pricing, all non-basic columns are scanned, but with partial pricing only a subset is scanned for every iteration. This can speed up several models
    "PRICE_LOOPLEFT": 1024,	#Scan entering/leaving columns left rather than right
    "PRICE_LOOPALTERNATE": 2048,	#Scan entering/leaving columns alternatingly left/right
    "PRICE_HARRISTWOPASS": 4096,	#Use Harris' primal pivot logic rather than the default
    "PRICE_TRUENORMINIT": 16384	#Use true norms for Devex and Steepest Edge initializations
}

scale_limit = {
    "lower":1,
    "higher":10,
    "num":10
}

#scaling can by any of the following values:

scaling_1 = {
    "SCALE_NONE": 0,	#No scaling
    "SCALE_EXTREME": 1,	#Scale to convergence using largest absolute value
    "SCALE_RANGE": 2,	#Scale based on the simple numerical range
    "SCALE_MEAN": 3,	#Numerical range-based scaling
    "SCALE_GEOMETRIC": 4,	#Geometric scaling
    "SCALE_CURTISREID": 7,	#Curtis-reid scaling
}

#Additionally, the value can be OR-ed with any combination of one of the following values:

scaling_2 = {
    "SCALE_QUADRATIC": 8,	 
    "SCALE_LOGARITHMIC": 16,	#Scale to convergence using logarithmic mean of all values
    "SCALE_USERWEIGHT": 31,	#User can specify scalars
    "SCALE_POWER2": 32,	#also do Power scaling
    "SCALE_EQUILIBRATE": 64,	#Make sure that no scaled number is above 1
    "SCALE_INTEGERS": 128,	#also scaling integer variables
    "SCALE_DYNUPDATE": 256,	#dynamic update
    "SCALE_ROWSONLY": 512,	#scale only rows
    "SCALE_COLSONLY": 1024	#scale only columns
}


simplex_type = {
    "SIMPLEX_PRIMAL_PRIMAL": 5,	#Phase1 Primal, Phase2 Primal
    "SIMPLEX_DUAL_PRIMAL": 6,	#Phase1 Dual, Phase2 Primal
    "SIMPLEX_PRIMAL_DUAL": 9,	#Phase1 Primal, Phase2 Dual
    "SIMPLEX_DUAL_DUAL": 10	#Phase1 Dual, Phase2 Dual
}



# In[151]:


### Generate LP solve params ###

#Comment out parameters you don't want to set here
#Parameters are added like this:  genLPSolveParam([param_name], ([sub_param_name], [sub_param_type], [sub_param_values]))
#For parameters with no "sub-parameters", just add the parameter name, parameter type and parameter values as a single "sub-parameter"

#The formatting of the code is located in a dict in the function genLPSolveParamSet()
#This formatting will determine if how your sub parameters should be place in the function e.g. "{} + {}" 

pList = []

addToPList("anti_degen",("antidegen", CATEGORIAL_COMBIN, antidegen))
addToPList("basiscrash",("basis_crash", CATEGORIAL_SINGLE_VAL, basiscrash))
addToPList("bb_depthlimit",("bb_depth_absolute", BOOL_SINGLE_VAL, bb_depth_absolute), ("bb_depthlimit", LIN_RANGE_INT,bb_depth_limit))
addToPList("bb_rule",("bb_rule_1", CATEGORIAL_SINGLE_VAL, bb_rule_1), ("bb_rule_2", CATEGORIAL_COMBIN, bb_rule_2))
addToPList("epsb", ("eps_b", LOG_RANGE_FLOAT, eps_b))
addToPList("epsd", ("eps_d", LOG_RANGE_FLOAT, eps_d))
addToPList("epsel", ("eps_el", LOG_RANGE_FLOAT, eps_el))
addToPList("epsperturb", ("eps_perturb", LOG_RANGE_FLOAT, eps_perturb))
addToPList("epspivot",("eps_pivot", LOG_RANGE_FLOAT, eps_pivot))
addToPList("improve",("improve", CATEGORIAL_COMBIN, improve))
#addToPList("infinite",("infinite", LOG_RANGE_FLOAT, infinite))
addToPList("maxpivot",("max_pivot", LIN_RANGE_INT, max_pivot))
addToPList("mip_gap", ("mip_gap_absolute", BOOL_SINGLE_VAL, mip_gap_absolute), ("mip_gap", LOG_RANGE_FLOAT, mip_gap))
addToPList("presolve", ("presolve", CATEGORIAL_COMBIN, presolve))
addToPList("pivoting", ("pivoting_1", CATEGORIAL_SINGLE_VAL, pivoting_1), ("pivoting_2", CATEGORIAL_SINGLE_VAL, pivoting_2))
addToPList("scalelimit", ("scale_limit", LOG_RANGE_INT, scale_limit))
addToPList("scaling", ("scaling_1", CATEGORIAL_SINGLE_VAL, scaling_1), ("scaling_2", CATEGORIAL_COMBIN, scaling_2))
addToPList("simplextype", ("simplex_type", CATEGORIAL_SINGLE_VAL, simplex_type))


# In[152]:


# Copy and paste the printed code into lpsolve.py after done, this will set the parameters you selected to be set

print(genLPSolveParamSet(pList))


# In[128]:


# These are the hyper-values, key-map and parameters lists respectively

full_stack = genParamKeyValues(pList)

hyper_vals = full_stack[0]
key_map = full_stack[1]
parameters = full_stack[2]
type_list = full_stack[3]


# In[124]:
# lpsolve_config template
lpsolve_config_content = (
"from numpy import *\n\n"
"dimension = {}\n".format(len(parameters)) +
"hyper_values = {}\n".format(hyper_vals) +
"key_map = {}\n".format(key_map) + 
"parameters = {}\n".format(parameters))

f = open("lpsolve_config.py", "w")
f.write(lpsolve_config_content)
f.close()

# In[125]:


#Test generated hyper-values

print(genFaux(hyper_vals))


# In[126]:


#For the markdown

def tablizer(keyMap, typeList, params):
    header = " __Parameter Name__ | __Type__ | __Default__ | __Min__ | __Max__ "
    headerSeparator = "--- | --- | --- | --- | --- "
    
    print(header)
    print(headerSeparator)
    
    for param in params:
        print("{} | {} |  | {} | {}".format(param, typeList[params.index(param)], keyMap[params.index(param)][0.0], keyMap[params.index(param)][1.0]))


# In[127]:


tablizer(key_map, type_list, parameters)


# In[ ]:





# In[188]:


import re

test = re.findall(r"\d*\.\d*(?:e[\+|-]{0,1}\d+){0,1}", "(.1)")
print(float(test[0]))


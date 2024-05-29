from typing import Dict, Any, Union, Callable


def eval_constraint_str(para: Dict[str, Any], constr: str) -> bool:
    """ Constraint evaluation: if the output is lower or equal to 0 then the input is valid """
    d = {}
    exec("d = {}", d)
    for k, v in para.items():
        exec(f"d['{k}'] = {v}", d)
    exec(f"aux = {constr}", d)
    return d["aux"]


def eval_constraint(para: Dict[str, Any], constr: Union[Callable[[Dict[str, Any]], bool], str]) -> float:
    """ Constraint evaluation: if the output is lower or equal to 0 then the input is valid """
    if isinstance(constr, str):
        return eval_constraint_str(para=para, constr=constr)
    return constr(para)

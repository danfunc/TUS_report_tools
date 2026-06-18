from __future__ import annotations
import argparse as args
import math
import re

SUM_RE = re.compile(r'^(sum|prod)_\(([A-Za-z_]\w*)=(.+)\)\^(.+)\((.+)\)$')


def parse_sum_prod(expr: str):
    expr = expr.strip()

    if not (expr.startswith("sum_(") or expr.startswith("prod_(")):
        return None

    kind = "sum" if expr.startswith("sum_(") else "prod"

    p1 = expr.index("(")
    p2 = find_matching_paren(expr, p1)

    header = expr[p1 + 1:p2]

    if "=" not in header:
        raise ValueError("invalid summation header")

    var, lower = header.split("=", 1)

    if p2 + 1 >= len(expr) or expr[p2 + 1] != "^":
        raise ValueError("missing upper bound")

    body_start = expr.find("(", p2 + 2)

    upper = expr[p2 + 2:body_start]

    body_end = find_matching_paren(expr, body_start)

    if body_end != len(expr) - 1:
        raise ValueError("unexpected trailing tokens")

    body = expr[body_start + 1:body_end]

    return kind, var.strip(), lower.strip(), upper.strip(), body.strip()

def collect_names(expr):
    names = set(re.findall(r"\b[A-Za-z_]\w*\b", expr))

    names -= bound_variables(expr)

    reserved = {
        "sum",
        "prod",
        "math",
        "range",
        "int",
    }

    return sorted(names - reserved)

def bound_variables(expr: str):
    result = set()

    def walk(s):
        parsed = parse_sum_prod(s)
        if not parsed:
            return

        _, var, lower, upper, body = parsed

        result.add(var)

        walk(lower)
        walk(upper)
        walk(body)

    walk(expr)
    return result

def translate_expr(expr: str):
    expr = expr.strip()

    parsed = parse_sum_prod(expr)

    if parsed:
        kind, var, lower, upper, body = parsed

        lower_py = translate_expr(lower)
        upper_py = translate_expr(upper)
        body_py = translate_expr(body)

        if kind == "sum":
            return (
                f"sum({body_py} "
                f"for {var} in range(int({lower_py}), int({upper_py}) + 1))"
            )

        return (
            f"math.prod({body_py} "
            f"for {var} in range(int({lower_py}), int({upper_py}) + 1))"
        )

    return expr.replace("^", " ** ")

def find_matching_paren(s: str, start: int) -> int:
    depth = 1
    for i in range(start + 1, len(s)):
        if s[i] == "(":
            depth += 1
        elif s[i] == ")":
            depth -= 1
            if depth == 0:
                return i
    raise ValueError("unmatched parenthesis")


def typst_math_to_lambda_source(expr: str, params=None):
    body = translate_expr(expr)
    if params is None:
        params = collect_names(expr)
    return f"lambda {', '.join(params)}: {body}"

def compile_typst_math(expr: str, params=None):
    src = typst_math_to_lambda_source(expr, params)
    return eval(src, {'math': math}), src

def main():
    parser = args.ArgumentParser()

    parser.add_argument("expr")
    parser.add_argument(
        "-p",
        "--params",
        nargs="*",
        default=None,
    )

    parsed = parser.parse_args()

    src = typst_math_to_lambda_source(
        parsed.expr,
        parsed.params
    )

    vars_ = (
        parsed.params
        if parsed.params is not None
        else collect_names(parsed.expr)
    )
    
    print("=" * 50)
    print("【Typst Expression】")
    print(" ", parsed.expr)
    print("-" * 50)
    print("【Variables】 (Auto-Detect)")
    print(" ", vars_)
    print("-" * 50)
    print("【Python Lambda】")
    print(" ", src)
    print("=" * 50)
    
    
    
if __name__ == '__main__':
    main()


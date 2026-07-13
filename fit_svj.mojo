from affine_svj import (
    affine_svj_fit_usage,
    parse_affine_svj_fit_args,
)
from affine_svj.fit_command import run_affine_svj_fit
from std.collections import List
from std.sys import argv


def main() raises:
    var raw_args = argv()
    var args = List[String]()
    for i in range(1, len(raw_args)):
        args.append(String(raw_args[i]))

    try:
        var config = parse_affine_svj_fit_args(args)
        if config.show_help:
            print(affine_svj_fit_usage())
            return
        run_affine_svj_fit[8](config)
    except error:
        print("fit_svj:", error)
        print(affine_svj_fit_usage())
        raise error^

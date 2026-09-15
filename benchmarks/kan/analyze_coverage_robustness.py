import argparse
import math
import tomllib
from pathlib import Path

from analyze_undersmoothing import analyze


CONDITIONS={
    "reference":dict(step=0.25,sigma=0.015,response="nonlinear"),
    "sparse":dict(step=0.5,sigma=0.015,response="nonlinear"),
    "noisy":dict(step=0.25,sigma=0.03,response="nonlinear"),
    "localized":dict(step=0.25,sigma=0.015,response="localized"),
}


def truth(case,x):
    if case not in CONDITIONS:
        raise ValueError("unknown robustness condition")
    multiplier=(1+0.5*math.exp(-((x-0.9)/0.25)**2)) if case=="localized" else (1+0.35*math.sin(2*x))
    return 0.9*(1-x/2)*multiplier


def validate_design(path):
    design=tomllib.loads((path/"design.toml").read_text())
    assert design["study_stage"]=="robustness"
    assert not set(design["seeds"]) & set(range(3001,3051))
    for case in design["cases"]:
        expected=CONDITIONS[case]
        actual=design["scenarios"][case]
        assert actual["response"]==expected["response"]
        assert actual["observation_step"]==expected["step"]
        assert actual["noise_sigma"]==design["noise_sigma"][case]==expected["sigma"]
        assert actual["localized_centre"]==0.9 and actual["localized_width"]==0.25
        assert actual["localized_amplitude"]==0.5
        times=[i*expected["step"] for i in range(round(5/expected["step"])+1)]
        assert actual["observation_times"]==times
        for seed in design["seeds"]:
            observed=tomllib.loads((path/"datasets"/f"{case}__{seed}.toml").read_text())
            assert observed["times"]==times
            assert observed["data_seed"]==actual["data_seed_base"]+seed
            assert len(observed["values"])==2
            assert all(len(column)==len(times) for column in observed["values"])
            for model in design["models"]:
                record=tomllib.loads((path/"replicates"/f"{case}__{model}__{seed}.toml").read_text())
                assert record["bootstrap_seed"]==actual["bootstrap_seed_base"]+seed
    return design


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--inputs",default=",".join(
        f"benchmarks/kan/results/robustness-{model}-{fraction}"
        for model in ("spline8","kan12_free") for fraction in ("1","quarter")))
    parser.add_argument("--output",type=Path,default=Path("benchmarks/kan/results/coverage-robustness-analysis"))
    args=parser.parse_args()
    paths=[Path(p).resolve() for p in args.inputs.split(",")]
    for path in paths:
        validate_design(path)
    analyze(paths,args.output,truth_function=truth,extra_scripts=(Path(__file__),))


if __name__=="__main__":
    main()

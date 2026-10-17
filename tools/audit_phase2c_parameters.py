"""Arithmetic-only counterfactual audit. Does not instantiate any FLT module."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]


def main():
    d,h,e,dec,s,n = 512,2048,12,7,49408,308
    ff = lambda width: 2*d*width+width+d
    mha = 4*(d*d+d)
    vim = 4*d*d
    baseline = s*d+2*n*d+e*(mha+ff(h)+vim+2*d)+dec*(2*mha+ff(h)+vim)
    visual = 41134277
    assert baseline == 112758272
    rows = []
    def add(category,name,delta):
        rows.append({"category":category,"counterfactual":name,"classification":"ASSUMED_UNAPPROVED",
                     "delta_from_baseline":delta,"flt":baseline+delta,"visual_plus_flt":visual+baseline+delta})
    for width in (512,1024,1536,2048,4096):
        add("ffn",f"two_affine_all_blocks_h{width}",(e+dec)*(ff(width)-ff(h)))
    for name,delta in (
        ("single_affine_all",(e+dec)*(d*d+d-ff(h))),
        ("single_affine_LD_only",dec*(d*d+d-ff(h))),
        ("LE2048_LD1024",dec*(ff(1024)-ff(h))),
        ("LE1024_LD2048",e*(ff(1024)-ff(h))),
        ("gated_three_affine_all_h2048",(e+dec)*(3*d*h+2*h+d-ff(h))),
        ("two_affine_without_bias",-(e+dec)*(h+d)),
    ): add("ffn",name,delta)
    for name,count in (("no_affine_LN_parameters",0),("baseline_LE1_LD0",12),
                       ("LE2_LD3",45),("LE3_LD4",64),("baseline_plus_two_final_LN",14)):
        add("normalization",name,(count-12)*2*d)
    for name,delta in (("ordinary_QKV_no_bias",-26*3*d),("ordinary_QKVO_no_bias",-26*4*d),
                       ("VIM_QKVO_add_bias",19*4*d),("vocab_output_add_bias",s)):
        add("bias",name,delta)
    for name,count in (("independent19",19),("LEshared_LDindependent",8),("LEindependent_LDshared",13),
                       ("one_per_branch",2),("one_global",1)):
        add("vim_sharing",name,(count-19)*vim)
    add("vocabulary","untied_output",s*d)
    report = {"method":"arithmetic only; no FLT instantiated", "baseline_flt":baseline,
              "accepted_visual":visual,"baseline_visual_plus_flt":baseline+visual,
              "paper_reported_approximate":141000000,"difference_before_remaining_heads":baseline+visual-141000000,
              "counterfactuals":rows,
              "hypothetical_remaining_biased_adapter_classifier_and_temperature":(d*d+d)+(2*d+2)+1}
    (ROOT/"artifacts/phase2c1/parameter_sensitivity.json").write_text(json.dumps(report,indent=2)+"\n")
    for row in rows:
        print(row["category"],row["counterfactual"],row["delta_from_baseline"],row["visual_plus_flt"])


if __name__ == "__main__":
    main()

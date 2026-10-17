# Phase 2C parameter sensitivity audit

2026-10-11. **Arithmetic analysis, not architecture approval or implementation.** Only the Phase 2C.1 prompt/tokenizer and standalone singleton VIM are implemented. The 12-block LE, seven-block LD, FFNs, language normalization and vocabulary embedding/output remain unimplemented and unapproved.

The paper source is [MFVLR.pdf](../papers/MFVLR.pdf), §III-D pp. 6–7, §IV-A p. 8, and Tables IX/XV pp. 14–15. Original equations remain in [the reference analysis](phase2c_vim_analysis.md). Executable arithmetic and complete machine-readable rows are [audit_phase2c_parameters.py](../tools/audit_phase2c_parameters.py) and [parameter_sensitivity.json](../artifacts/phase2c1/parameter_sensitivity.json). That utility never constructs a model.

## Evidence and baseline

**EXPLICIT:** E=12, D=7, d=512, n=308, s=49,408 (§IV-A p. 8); LE pre-self-attention LN and FF prose (§III-D pp. 6–7); LD operation order, Eqs. (13)–(16); d×d VIM projections, Eqs. (4)–(11); reuse of Wvoc at the output, Eq. (24). The paper reports 141.00M parameters (Tables IX/XV), but does not establish an exact integer count or sufficiently precise counting scope.

**ASSUMED, still unapproved for full FLT:** two biased FF affine maps with h=2048; ordinary attention biases; 19 independent VIM allocations; one affine LN per LE block and none in LD; independent learned Pe/Pd; no output bias. The standalone VIM now has approved bias-free 512×512 Q/K/V/O, but that approval does **not** allocate it to 19 blocks or approve any LE/LD structure.

All table deltas below are **DERIVED conditional on the stated counterfactual assumptions**, holding every other baseline choice fixed. Alternative choices remain ASSUMED / PROPOSED; no tensor size was changed to match a paper total.

| Baseline component | Count |
|---|---:|
| Accepted MVE | 32,078,912 |
| Accepted VD | 9,055,365 |
| Accepted visual subtotal | 41,134,277 |
| Hypothetical tied Wvoc, 49408×512 | 25,296,896 |
| Hypothetical Pe+Pd, 2×308×512 | 315,392 |
| Hypothetical 12 LE blocks | 50,399,232 |
| Hypothetical 7 LD blocks | 36,746,752 |
| Hypothetical FLT subtotal | 112,758,272 |
| **Visual + hypothetical FLT** | **153,892,549** |
| Difference from reported 141,000,000, before remaining heads | **+12,892,549** |

The actual implemented standalone VIM contains 1,048,576 unique parameters, including 524,288 Q/K parameters with zero singleton-attention task gradients. This is not an implemented full FLT count. The actual visual modules plus **one** independently owned VIM would total 42,182,853. Tokenization and prompt construction have zero learned parameters; their BPE vocabulary is not Wvoc.

## FF structure and width

For two biased affine maps d→h→d, `P_FF(h)=2dh+h+d=1025h+512`. At h=2048 this is 2,099,712 per block. Across all 19 hypothetical blocks, each additional hidden unit adds 19,475 parameters. With separate widths, each LE hidden unit adds 12,300 and each LD hidden unit adds 7,175. Activation selection does not change this count, although it changes behavior.

The paper describes an FF network with a fully connected layer but does not give its expansion, number of affine maps or nonlinearity (§III-D p. 7). These alternatives therefore cannot be distinguished by its wording alone.

| Counterfactual FF choice | Delta vs baseline | Visual+FLT total |
|---|---:|---:|
| Two affine, h=512 everywhere | −29,913,600 | 123,978,949 |
| Two affine, h=1024 everywhere | −19,942,400 | 133,950,149 |
| Two affine, h=1536 everywhere | −9,971,200 | 143,921,349 |
| Two affine, h=2048 everywhere, baseline | 0 | 153,892,549 |
| Two affine, h=4096 everywhere | +39,884,800 | 193,777,349 |
| LE h=2048, LD h=1024 | −7,347,200 | 146,545,349 |
| LE h=1024, LD h=2048 | −12,595,200 | 141,297,349 |
| One biased affine 512→512 in every block | −34,904,064 | 118,988,485 |
| One biased affine 512→512 in LD only | −12,859,392 | 141,033,157 |
| Gated three-affine FF, h=2048 everywhere | +19,961,856 | 173,854,405 |
| Baseline two-affine FF with all FF biases removed | −48,640 | 153,843,909 |

The single-affine count is `d²+d=262,656`. The gated example uses two separate biased d→h branches and one biased h→d output, `3dh+2h+d=3,150,336`; it is a sensitivity example, not a published or recommended MFVLR feature.

The LD-single-affine case happens to lie near 141M. That is **not evidence** that the authors used it. Its total is still 33,157 above 141,000,000 before the omitted adapter/classifier/temperature. Assuming biased 512→512 and 512→2 maps plus one scalar would add another 263,683, making this counterfactual 141,296,840. Those heads are not implemented or approved; their bias choices are themselves assumptions. Approximate paper counts, omitted ownership details and compensating choices make inverse reconstruction from a total underdetermined.

## Normalization

An affine LayerNorm at d=512 adds 1,024 parameters; epsilon, pre/post placement and running order do not change that number. Their mathematical effects can be substantial despite identical counts. The baseline has 12 such layers (12,288 parameters). The LE LN is explicit in §III-D p. 6; the absence of other norms is an engineering interpretation of incomplete equations, not proof of the authors' implementation.

| Counterfactual normalization allocation | Delta | Visual+FLT total |
|---|---:|---:|
| Existing 12 LNs made non-affine | −12,288 | 153,880,261 |
| Baseline: one LE LN, no LD LN | 0 | 153,892,549 |
| Two LN per LE and three per LD: 45 affine LN total | +33,792 | 153,926,341 |
| Three LN per LE and four per LD: 64 total | +53,248 | 153,945,797 |
| Baseline plus one final LE LN and one final LD LN | +2,048 | 153,894,597 |

Removing the 12 LNs entirely gives the same parameter delta as making them non-affine, but discards an explicitly described LE operation and is not a faithful reference alternative. Adding LD norms or moving norms would require a new contract even if the count were unchanged. These tens-of-thousands changes do not resolve a 12.89M gap by themselves.

## Projection biases

The hypothetical FLT has 26 ordinary attention modules: 12 LE self-attention, seven LD self-attention and seven LD memory-attention. Each Q/K/V/O bias has 512 elements. Projection bias choices are not specified by the ordinary-MHA equations. Phase 2C.1's VIM bias-free choice is approved for the primitive, and any VIM bias addition below is counterfactual only.

| Counterfactual | Delta | Visual+FLT total |
|---|---:|---:|
| Remove ordinary Q/K/V biases, retain O biases | −39,936 | 153,852,613 |
| Remove all ordinary Q/K/V/O biases | −53,248 | 153,839,301 |
| Add Q/K/V/O biases to all 19 VIM instances | +38,912 | 153,931,461 |
| Add vocabulary output bias | +49,408 | 153,941,957 |

Eq. (24) displays no vocabulary output bias. Adding one would change that reference formulation. VIM Q/K bias gradients would also be zero under finite singleton key-axis attention; this does not justify omitting their count if they were registered parameters. None of these changes is implemented.

## VIM independence versus sharing across depth

One bias-free VIM has `4d²=1,048,576` parameters. Subscripted block notation and Fig. 4 motivate independent instances, but the paper does not explicitly specify depth sharing. All cases below share the **input visual tensor**; that is distinct from sharing projection parameters.

| Hypothetical allocation | Unique VIM instances | VIM subtotal | Delta | Visual+FLT total |
|---|---:|---:|---:|---:|
| Independent 12 LE + 7 LD, baseline | 19 | 19,922,944 | 0 | 153,892,549 |
| One shared LE VIM + seven independent LD | 8 | 8,388,608 | −11,534,336 | 142,358,213 |
| Twelve independent LE + one shared LD | 13 | 13,631,488 | −6,291,456 | 147,601,093 |
| One shared within each branch | 2 | 2,097,152 | −17,825,792 | 136,066,757 |
| One shared across both branches and all depths | 1 | 1,048,576 | −18,874,368 | 135,018,181 |

Sharing changes update accumulation and capacity. Neither constant singleton weights nor zero Q/K task gradients makes independent V/O projections redundant across nonlinear blocks. Do not delete Q/K: in the 19-instance hypothetical model this would remove 9,961,472 elements and change the registered graph/optimizer behavior, contrary to the approved reference primitive. The implementation retains all four matrices and explicitly computes attention.

## Vocabulary tying

The paper's Eq. (24), `Tpre = Trec^d Wvoc^T`, explicitly reuses the vocabulary matrix introduced in §III-D p. 6. Under that reference interpretation the trainable matrix is counted once, despite serving multiple lookup/output roles.

| Counterfactual | Delta | Visual+FLT total |
|---|---:|---:|
| Tied vocabulary, baseline | 0 | 153,892,549 |
| Independent 49408×512 output matrix | +25,296,896 | 179,189,445 |

Untying is a departure from Eq. (24), not a faithful way to repair count mismatch. It increases the gap. A trainable bias would be separate as shown above. A tokenizer with 49,408 entries has no learned matrix in Phase 2C.1 and must not be counted as 25,296,896 implemented parameters.

## Interpretation and next review

FF structure/width and VIM sharing dominate uncertainty; normalization and bias choices are comparatively small. Distinct architectures can have similar totals, and rearranging norms can leave totals identical. Matching 141.00M would not establish faithful architecture, training behavior or performance. The full-model count scope, exact LE/LD FF definitions, normalization placement, projection biases and depth sharing require author evidence or explicit engineering approval.

Retain the accepted visual topology and the approved standalone VIM unchanged. This audit recommends no dimensional adjustment. Full FLT design remains PROPOSED, with its discrepancy visible for the next review.

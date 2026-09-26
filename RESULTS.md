# Results


### Main runs (50M tokens each)

| run | arch | batch | steps | tokens | final train (EMA) | val loss | val ppl | tok/s | peak alloc MB | peak resv MB | wall min |
|---|---|---|---|---|---|---|---|---|---|---|---|
| r1b_baseline_b16_lr0.00075 | baseline | 16 | 6103 | 50.0M | 3.6489 | 3.7827 | 43.9 | 40.5k | 2130 | 2348 | 20.6 |
| r2_rev_mid_b16 | rev_mid | 16 | 6103 | 50.0M | 3.6595 | 3.7968 | 44.6 | 28.8k | 1157 | 1482 | 28.9 |
| r3_rev_mid_b192_lr0.0013 | rev_mid | 192 | 508 | 49.9M | 4.1395 | 4.2545 | 70.4 | 35.2k | 4502 | 5118 | 23.6 |
| r1_baseline_b16 | baseline | 16 | 6103 | 50.0M | 4.0163 | 4.1503 | 63.5 | 33.8k | 2130 | 2348 | 24.7 |
| r3_rev_mid_b192_lr0.00065 | rev_mid | 192 | 508 | 49.9M | 4.3731 | 4.4691 | 87.3 | 34.2k | 4502 | 5118 | 24.3 |
| r3_rev_mid_b192_lr0.0026 | rev_mid | 192 | 508 | 49.9M | 4.3337 | 4.4020 | 81.6 | 34.0k | 4502 | 5118 | 24.5 |
| r3_rev_mid_b192_lr0.0052 | rev_mid | 192 | 508 | 49.9M | 4.9160 | 4.9890 | 146.8 | 32.2k | 4502 | 5118 | 25.9 |


### Variant + LR selection, 10M tokens, batch 16 (AFTER the autocast-cache fix)

| run | arch | batch | steps | tokens | final train (EMA) | val loss | val ppl | tok/s | peak alloc MB | peak resv MB | wall min |
|---|---|---|---|---|---|---|---|---|---|---|---|
| s2_baseline_0.003 | baseline | 16 | 1220 | 10.0M | 5.1148 | 5.1812 | 177.9 | 49.7k | 2130 | 2348 | 3.4 |
| s3_baseline_0.0002 | baseline | 16 | 1220 | 10.0M | 5.1023 | 5.1690 | 175.7 | 48.9k | 2130 | 2348 | 3.4 |
| s3_baseline_0.0004 | baseline | 16 | 1220 | 10.0M | 4.8415 | 4.9115 | 135.8 | 53.2k | 2130 | 2348 | 3.1 |
| s3_baseline_0.00075 | baseline | 16 | 1220 | 10.0M | 4.6766 | 4.7538 | 116.0 | 46.9k | 2130 | 2348 | 3.6 |
| s3_baseline_0.0015 | baseline | 16 | 1220 | 10.0M | 4.8795 | 4.9521 | 141.5 | 41.0k | 2130 | 2348 | 4.1 |
| s3_rev_euler_0.0002 | rev_euler | 16 | 1220 | 10.0M | 5.0780 | 5.1468 | 171.9 | 31.6k | 1154 | 1466 | 5.3 |
| s3_rev_euler_0.0004 | rev_euler | 16 | 1220 | 10.0M | 4.8482 | 4.9206 | 137.1 | 27.7k | 1154 | 1466 | 6.0 |
| s3_rev_euler_0.00075 | rev_euler | 16 | 1220 | 10.0M | 4.9093 | 4.9802 | 145.5 | 31.5k | 1154 | 1466 | 5.3 |
| s3_rev_euler_0.0015 | rev_euler | 16 | 1220 | 10.0M | 5.0126 | 5.0811 | 161.0 | 37.2k | 1154 | 1466 | 4.5 |
| s3_rev_euler_0.003 | rev_euler | 16 | 1220 | 10.0M | 5.2041 | 5.2686 | 194.1 | 32.0k | 1154 | 1466 | 5.2 |
| s3_rev_euler_0.006 | rev_euler | 16 | 1220 | 10.0M | 5.8443 | 5.8937 | 362.7 | 30.2k | 1154 | 1466 | 5.5 |
| s3_rev_mid_0.0004 | rev_mid | 16 | 1220 | 10.0M | 4.7768 | 4.8526 | 128.1 | 28.7k | 1157 | 1482 | 5.8 |
| s3_rev_mid_0.00075 | rev_mid | 16 | 1220 | 10.0M | 4.5872 | 4.6716 | 106.9 | 30.1k | 1157 | 1482 | 5.5 |
| s3_rev_mid_0.0015 | rev_mid | 16 | 1220 | 10.0M | 4.7699 | 4.8455 | 127.2 | 27.7k | 1157 | 1482 | 6.0 |
| s3_rev_mid_seed_0.00075 | rev_mid_seed | 16 | 1220 | 10.0M | 4.5869 | 4.6731 | 107.0 | 27.2k | 1176 | 1482 | 6.1 |


### Superseded: same sweeps BEFORE the fix (block weights were frozen)

| run | arch | batch | steps | tokens | final train (EMA) | val loss | val ppl | tok/s | peak alloc MB | peak resv MB | wall min |
|---|---|---|---|---|---|---|---|---|---|---|---|
| buggy_r2_rev_euler_b16 | rev_euler | 16 | 6103 | 50.0M | 4.9282 | 5.0565 | 157.0 | 33.8k | 1010 | 1194 | 24.7 |
| buggy_r3_rev_euler_b192_lr1.2e-2 | rev_euler | 192 | 508 | 49.9M | 5.0131 | 5.1193 | 167.2 | 31.3k | 4036 | 4878 | 26.6 |
| buggy_r3_rev_euler_b192_lr2.4e-2 | rev_euler | 192 | 508 | 49.9M | 4.9899 | 5.0975 | 163.6 | 34.4k | 4036 | 4878 | 24.3 |
| buggy_s2_rev_euler_0.0015 | rev_euler | 16 | 1220 | 10.0M | 5.2244 | 5.3073 | 201.8 | 40.4k | 1010 | 1194 | 4.1 |
| buggy_s2_rev_euler_0.003 | rev_euler | 16 | 1220 | 10.0M | 5.1860 | 5.2715 | 194.7 | 39.1k | 1010 | 1194 | 4.3 |
| buggy_s2_rev_euler_0.006 | rev_euler | 16 | 1220 | 10.0M | 5.1799 | 5.2674 | 193.9 | 37.9k | 1010 | 1194 | 4.4 |
| buggy_s2_rev_mid_0.0015 | rev_mid | 16 | 1220 | 10.0M | 5.2230 | 5.3044 | 201.2 | 35.3k | 1011 | 1210 | 4.7 |
| buggy_s2_rev_mid_0.003 | rev_mid | 16 | 1220 | 10.0M | 5.1992 | 5.2830 | 197.0 | 32.5k | 1011 | 1210 | 5.1 |
| buggy_s2_rev_mid_0.006 | rev_mid | 16 | 1220 | 10.0M | 5.1955 | 5.2795 | 196.3 | 33.8k | 1011 | 1210 | 4.9 |


### GPU clock behaviour (throttling)

| log | samples>50% util | peak SM MHz | first-10 mean | last-50 mean | median | max temp C |
|---|---|---|---|---|---|---|
| clocks_final.log | 457 | 1815 | 1429 | 860 | 765 | 89 |
| clocks_postfix.log | 1383 | 1800 | 1372 | 909 | 885 | 89 |
| clocks_r1_final.log | 217 | 1882 | 1681 | 826 | 825 | 89 |
| clocks_r2.log | 55 | 1447 | 1224 | 889 | 877 | 89 |
| clocks_run1.log | 227 | 1867 | 1572 | 804 | 712 | 89 |
| clocks_run2.log | 226 | 1860 | 1552 | 779 | 712 | 89 |

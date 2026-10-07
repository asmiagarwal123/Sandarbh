# AUTALIC dataset use and provenance

Version 1.1.0 — 2026-10-07

The local `AUTALIC.csv` has SHA-256 `66ccaa43f9f3d8694c4826d8b999a4e562041637cad3c8c396590fbe78d3d408`, size 834,207 bytes, six expected columns, and 2,400 rows. Its download URL and retrieval date are **UNKNOWN / NEEDS VERIFICATION**. **OFFICIAL BYTE-LEVEL MATCH UNKNOWN / NEEDS AUTHOR VERIFICATION.** Do not silently replace it.

Authoritative references are the ACL 2025 paper at <https://aclanthology.org/2025.acl-long.1022/>, arXiv 2410.16520, and <https://nrizvi.github.io/AUTALIC.html>. The paper describes 2,400 targets, 2,014 preceding contexts, 2,400 following contexts, 242 majority-positive examples, and mean Fleiss' kappa about 0.25. It also reports 242 positive plus 2,160 negative, an internally inconsistent total of 2,402.

The saved local audit finds 122 all-empty records, 2,278 eligible records, and 256 eligible positives under the project mapping: annotation 1 is positive; 0 and -1 map to other; at least two positive votes produce hard class 1. Local nonempty counts are target 2,278, preceding 1,876, following 2,278. A1/A2/A3 are positions in separate three-person annotation groups over three 800-item segments, not persistent annotator identities.

Use restrictions recorded from the official paper/project materials:

- Academic, scientific, and educational research is permitted under the stated conditions.
- Dataset redistribution is restricted.
- Commercial use requires written permission.
- Automated moderation requires prior author approval.
- Store the dataset securely; do not publicly expose sentences or artifacts that reconstruct them.
- The data is US/Reddit-contextual and should not be generalized without validation.

Author-contact draft (not sent):

> Subject: AUTALIC local-file clarification. We are auditing a research-only local copy (SHA-256 66ccaa43...). Its 2,400 rows include 122 rows with all three text fields empty; among 2,278 eligible rows, our documented binary mapping yields 256 majority-positive examples. Could you confirm whether the empty rows are expected and explain the paper's 242-positive figure (and the 242 + 2,160 = 2,402 total)? We will not redistribute the data.

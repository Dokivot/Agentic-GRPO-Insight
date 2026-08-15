# Evaluation Summary

**pass@1 (any success):** 0.180
**pass^1 (avg success):** 0.110
**pass^4 (stability):** 0.180
**pass^8 (stability):** 0.000
**Avg turns:** 10.08
**Avg tool calls:** 7.67
**Error rate:** 0.305
**Eval time:** 14664.1s

## Failure Mode Distribution

| Mode | Count | Percentage |
|------|-------|------------|
| tool_call_error | 0 | 0.0% |
| context_overflow | 0 | 0.0% |
| max_turns_exceeded | 26 | 14.6% |
| wrong_action | 123 | 69.1% |
| user_simulator_breakdown | 5 | 2.8% |
| premature_termination | 23 | 12.9% |
| loop | 1 | 0.6% |
| other | 0 | 0.0% |
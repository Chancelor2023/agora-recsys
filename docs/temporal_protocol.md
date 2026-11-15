# Temporal protocol V1
Status: agreed by Brett and Ronald in the team group, as reported by Jerry.
This records a methodological agreement; it does not certify that the data
pipeline has implemented or verified these rules.

## Time and split
Use half-open intervals [start, end):
| Partition | Start inclusive | End exclusive |
| --- | --- | --- |
| Train | 2023-11-23 00:00 | 2023-11-30 00:00 |
| Validation | 2023-11-30 00:00 | 2023-12-03 00:00 |
| Test | 2023-12-03 00:00 | 2023-12-08 00:00 |

Brett must verify the source timestamp semantics and timezone. Interpret these
calendar boundaries in that timezone before harmonising timestamps. Do not
silently assume UTC. In experiment.toml, source_timezone = "UNVERIFIED" is
an explicit blocker for producing trustworthy real-data labels.

## Profiles and learned quantities
For validation and test, freeze profiles from training reads only. Require at
least five distinct canonical articles read during training. Learn popularity,
scorer weights and all interaction-derived features from training only.
For a TRAINING feed at t, build its profile using reads strictly before t, not
the final end-of-training profile. An insufficient-history training-feed policy
still needs documenting by Ronald; the end-of-training eligibility rule alone
does not establish an eligible profile for an earlier feed.
Select parameters on validation and freeze them before evaluating test.

## First-read attribution
For each reader feed at t:
window_start = t
window_end = min(next feed time for this reader, t + 24 hours,
                 partition end, observation end)
Omit the next-feed bound when there is no next feed. Use [window_start, window_end).
A canonical candidate receives 1 only if its FIRST read occurs inside the window.
Otherwise it receives 0, a weak negative rather than confirmed rejection.
Do not assign reads to articles absent from the candidate list.
Do not reuse the same first-read event across adjacent feeds.
Compute the next-feed boundary from the original reader feed sequence, before
filtering feeds or topics, so filtering does not extend attribution windows.

Brett must verify actual observation end, final-feed handling and mapping of
duplicate IDs, URLs and experimental variants. Preserve source provenance.
Flag and count windows shortened by partition/observation boundaries; keep
the agreed V1 labels and optionally assess sensitivity to excluding these cases.

## Evaluation
Use identical cleaned candidate universes for all compared models. Keep observed
labels separate from model and reranker inputs. Provide complete candidate
rankings, not only the top 10/20.
Average NDCG@10 and Recall@20 by eligible feed. Count and exclude feeds with no
positive label from both means. Use @5 metrics for supplementary public affairs.
Expose candidate counts and explain that short lists can make recall trivial.
Evolve reranking exposure history chronologically, using separate state for each
algorithm. Do not use future labels to update exposure state.

## Handoff
Brett: canonical candidates, separate labels, eligible histories, and audit counts.
Ronald: complete scored rankings for exactly those feeds and candidates.
Jerry: strict join validation, per-feed metrics, aggregate output, exclusion counts.
See evaluation_handoff.md. The evaluator alone cannot verify temporal leakage.

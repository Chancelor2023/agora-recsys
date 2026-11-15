# Contributing
## Branches
Use one active branch per member and task:
- `jerry/evaluation-v1`
- `brett/data-pipeline-v1`
- `ronald/relevance-model-v1`
- `romain/reranking-v1`
- `mathis/api-v1`

Start from the latest main. Commit small, coherent changes, then push your branch and open a pull request into main. Ask one teammate to review. Integrate regularly rather than waiting until the end of the sprint. Create a fresh branch from main for the next task after merging.

## Shared files
Coordinate changes to experiment settings, interfaces, dependencies and README. Folder ownership indicates responsibility, not exclusive permission to contribute.

## Pull request evidence
Include the problem solved, exact run command, actual output, limitations and next action. Do not claim that the component works before running it.

## Data and evaluation
Keep future reads and evaluation labels out of profile, feature and reranking inputs. Preserve original data provenance and experimental flags. Generated recommendation lists are not proof that every candidate was viewed. Keep evaluation labels in a separate file.

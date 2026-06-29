# cusadi2

- We add another axes of parallelism to the original cusadi in the form of level-parallel parallelism
- We also added stage-parallel execution - but this seems to have been done in a way in the 2D formulation in cusadi proper - but we do it in a way that we only compile a single stage - dramatically decreasing compilation times for structured problems

# How does it work

Read files in this order:
1. cusadi_operations.py
2. kernel_template.py
3. dag.py
4. generate.py
5. cusadi2.h


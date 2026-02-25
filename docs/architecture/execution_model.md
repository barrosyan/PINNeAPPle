# Execution Model

1. ProblemDefinition defines equations and constraints.
2. Domain/Geometry produces sampling.
3. Model predicts fields from coords.
4. Physics/PINN computes residuals and builds loss terms.
5. Solver applies an optimization strategy.
6. Backend executes the run.
7. Researcher evaluates metrics and stores artifacts.

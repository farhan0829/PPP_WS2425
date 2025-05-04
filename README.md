# PPP_WS2425

# PDE Toolkit 🛠️

A compact, educational code‑base for solving elliptic partial‑differential equations (PDEs) in **1‑D** and **2‑D** with both Finite Difference and Finite Element discretisations, plus a **Genetic Algorithm (GA)** module that tackles inverse problems such as reconstructing unknown source terms.  

---

## Project Outline

| Layer | Folder | Key Files | Role |
|-------|--------|-----------|------|
| **Execution** | `run/` | `main.py` | CLI entry‑point; chooses FDM or FEM and launches simulations. |
| **Problem Setup** | `problems/` | `problem_setup.py` | Generates benchmark problems with analytic solutions *u<sub>d</sub>* and coefficients. |
| **Grid & BCs** | `core/` | `grid.py` | Creates structured grids; enforces Dirichlet BCs. |
| **Solver Dispatcher** | `core/` | `pde_solver.py` | Routes the call to FDM or FEM back‑ends. |
| **Numerical Kernels** | `toolkit/` | `fd_solver.py`, `fem_solver.py` | Assemble & solve linear systems for FDM / FEM. |
| **Fitness Layer** | `ga/` | `fitness_evaluator.py` | Computes error between numeric *u* and target *u<sub>d</sub>*. |
| **Evolution Engine** | `optimizer/` + `ga/` | `ga_optimizer.py`, `population.py`, `individual.py` | Implements GA (selection, crossover, mutation) to minimise fitness. |
| **Verification** | `testcases/` | `test_fd_solver.py`, `test_fd_solver_with_ga.py` | Convergence checks and GA validation in 1‑D & 2‑D. |
| **Visuals** | `figures/` | PNG/TikZ | Contours, error maps, flowcharts for the report. |

---

## Process Flow 

```mermaid
flowchart TD
  A(run/main.py) --> B[create_problem]
  B --> C[u_d / coeffs]
  B --> D[PDESolver]
  D -->|FDM| E1[assemble_fd → fd_solver]
  D -->|FEM| E2[assemble_fem → fem_solver]
  E1 --> F[FitnessEvaluator]
  E2 --> F
  F --> G[GAOptimizer]
  G --> H[Population & Individual]
  H -->|best fields| I[Updated coefficients]
  I --> D
  G --> J((Plots / Logs))

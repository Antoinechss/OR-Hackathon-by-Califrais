<h1 align="center">KIRO 2025 – Route Optimization Solver</h1>

<p align="center">
  <em>
    A fast heuristic solver for time-windowed vehicle routing with heterogeneous fleets and time-dependent travel times.
  </em>
</p>

<p align="center">
  <img width="900" alt="KIRO 2025 Solver Overview" src="https://github.com/user-attachments/assets/2be941b2-663f-40b0-97e5-141f0f64dfab" />
</p>

---

## Overview

This repository contains my solution to the <strong>KIRO 2025 Operations Research Hackathon</strong>, organized by <strong>Califrais</strong> and <strong>CERMICS</strong>.

The objective is to compute efficient delivery routes under realistic operational constraints, including vehicle heterogeneity, time windows, parking delays, time-dependent travel speeds, and multiple cost components.

The solver is fully hand-coded in Python and achieved a competitive score within the 10-minute runtime constraint imposed by the competition.

---

## Problem Description

Each solution must assign every order to exactly one route while satisfying:

- Vehicle capacity limits  
- Time windows (waiting allowed)  
- Time-dependent travel times  
- Routes start and end at the depot  
- Heterogeneous vehicle families (1, 2, and 3)  

### Objective Function

Minimize:

<strong>Total cost = rental cost + fuel cost + clustering (Euclidean radius) penalty</strong>

Additional details:

- Distances are computed using Manhattan and Euclidean metrics  
- Travel times depend on the time of day via a 4-term Fourier series  

---

## Methodology

### 1. Preprocessing

- Vectorized computation of Manhattan and Euclidean distance matrices  
- Distance computation from spherical latitude/longitude coordinates  
- Caching of time-dependent travel times (<code>travel_cache</code>) to reduce runtime  

### 2. Greedy Initial Solution

- Construct routes iteratively  
- Insert the nearest feasible next order  
- Evaluate all vehicle families and keep the best feasible choice  

### 3. Local Search Optimization (Relocation)

Relocation-based local search is applied until no improvement remains:

- Remove a node from its current route  
- Attempt reinsertion into another route  
- Apply a first-improving move strategy  

Implementation optimizations:

- Incremental arrival-time propagation  
- Nearest-neighbor insertion candidates  
- Early computation of fuel cost deltas  

---

## Output

- Generates <code>routesX.csv</code> files (KIRO-compatible format)  
- One solution file per instance  
- Files are written automatically to the current directory  

---

## Acknowledgements

KIRO 2025 Operations Research Hackathon
Organized by Califrais and CERMICS

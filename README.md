<H1> KIRO 2025 – Route Optimization Solver </H1>
A fast heuristic for time-windowed vehicle routing with heterogeneous fleet & time-dependent travel times.

This repository contains my solution for the KIRO 2025 Operations Research Hackathon organized by Califrais & CERMICS.
The goal is to compute efficient delivery routes under real constraints: vehicle capacities, time windows, parking time, Fourier-based speed variations, fuel & rental costs, and an Euclidian radius penalty.

Solver is fully hand-coded in Python and obtained a competitive score within the 10-minute runtime limit.

**Problem Summary**

Each solution must assign every order to exactly one route while respecting:

* Vehicle capacity

* Time windows (with waiting allowed)

* Time-dependent travel times

* Start and end at the depot

* Heterogeneous vehicle families 1, 2 and 3

Objective: minimize **total cost = rental + fuel + clustering penalty**

Distances are computed using Manhattan and Euclidean metrics.
Travel times depend on the time of day via a 4-term Fourier series.

**Approach**

1. Preprocessing

Vectorized computation of Manhattan & Euclidean distance matrices from spherical Lat and Long coordinates

Caching of travel times (travel_cache) to save computational speed 

2. Greedy Initial Solution

Build routes one by one

Always insert the nearest feasible next order

Try all vehicle families and keep the best solution

3. Local Search optimization (Relocation)

Remove a node from a route

Try inserting it into another route to get a better solution. Apply first improving move until no improvements remain

Uses:

* incremental arrival-time propagation

* nearest-neighbor insertion candidates

* early fuel-delta computing 

**Output**

Produces routesX.csv files (KIRO-compatible format), one per instance.

**How to Run ?**

Install dependencies
<pip install pandas numpy>

Run the solver
<python kiro.py>

CSV solutions will be generated automatically inside the current folder

Notes : Further Improvements such as 2-opt, cross-exchange, or metaheuristics could further improve solution quality.

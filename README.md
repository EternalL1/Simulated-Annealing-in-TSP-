# Simulated-Annealing-in-TSP-
Simulated Annealing (SA) is a probabilistic meta-heuristic inspired by annealing in metallurgy, where a material is heated and cooled slowly so it settles into a low-energy state.

For TSP, the "energy" is the total tour distance and the "temperature" T controls how willing the search is to accept worse tours.

Procedure (Algorithm 2.5 in the assigned paper)
1. Input the cities and the pairwise distance matrix.
2. Start with a random tour and compute its total distance.
3. Set a high initial temperature T.
4. Generate a new tour by a small change, such as swapping two cities.
5. Compute the new tour's distance.
6. If the new tour is shorter, accept it.
7. Otherwise accept it with probability P = e^(−Δ/T), where Δ = new cost − current cost.
8. Cool: T = αT, with 0 < α < 1.
9. Stop when T is low enough, there is no improvement, or the iteration limit is reached.
10. Output the shortest tour found.

Advantages and limitations

• Advantages: simple to implement; escapes local optima by occasionally accepting worse tours; works with asymmetric (directed) distances; scales to large instances where exact methods are too slow.

• Limitations: results depend on parameters (T₀, α, stopping rule); runs vary because it is random; it gives no proof of optimality; for very small problems exact methods are faster and certain.

• Complexity (from the paper): each iteration costs O(n) to evaluate a tour. With i = O(n²) iterations the total is O(n³).

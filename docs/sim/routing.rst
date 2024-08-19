Routing
========

The Routing component plays a pivotal role in managing network traffic by employing different route methods based on
the simulation's requirements:

- **NLI-Aware Routing**: Utilizes the least non-linear impairment (NLI) to guide route selection.
- **XT-Aware Routing**: Determines routes based on the least cross-talk (XT) interference.
- **Least Congested Routing**: Identifies routes with the lowest congestion level to optimize data transmission.
- **Shortest Path Routing**: Selects routes with the shortest physical distance, often measured by length.
- **K-Shortest Path Routing**: Finds the K shortest paths based on various criteria, allowing for increased flexibility in route selection.


.. automodule:: src.routing
    :members:
    :undoc-members:
    :private-members:

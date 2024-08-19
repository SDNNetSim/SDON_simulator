Spectrum Assignment
====================

The Spectrum Assignment component offers various methods for allocating spectrum resources within the simulation:

- **Forced Index Allocation**: Assigns spectrum resources based on user-specified indices, ensuring specific resource allocation.
- **Best Fit Allocation**: Utilizes the best-fit strategy to allocate spectrum resources, minimizing fragmentation and maximizing resource utilization.
- **First Fit Allocation**: Allocates spectrum resources using the first-fit strategy, where the first available spectrum block meeting the requirements is allocated.
- **Last Fit Allocation**: Allocates spectrum resources using the last-fit strategy, where the last available spectrum block meeting the requirements is allocated.
- **Priority-based Allocation**: Assigns spectrum resources based on priority criteria, such as priority-first or priority-last allocation methods.
- **Cross-Talk (XT) Aware Allocation**: Considers cross-talk interference when allocating spectrum resources, optimizing resource utilization and minimizing interference.


.. automodule:: src.spectrum_assignment
    :members:
    :undoc-members:
    :private-members:

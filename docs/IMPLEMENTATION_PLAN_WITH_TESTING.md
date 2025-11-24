# Implementation Plan: Topology Switching with Incremental Testing

**Date:** November 23, 2025  
**Goal:** Implement and test topology switching incrementally, ensuring each component works before moving to the next

**Revision:** Type 1 and Type 2 switches are tested together from Phase 1, since Type 2 is implemented via Type 1 (it just selects which VP to move based on triple point position). Real simulation data shows Type 2 switches can occur without Type 1 switches, confirming they should be tested as a unified mechanism.

---

## Overview: 5-Phase Incremental Plan

| Phase | Goal | Expected Outcome | Test Strategy |
|---|---|---|---|
| **0** | Verify dependencies | All imports work | Import test |
| **1** | Test both Type 1 & Type 2 switching | VPs move, triple points migrate | Isolated switch test |
| **2** | Fix Issue 1 | `triangle_segments` rebuilt | Post-switch rebuild |
| **3** | Test switching + calculations | Area/perimeter work after switches | Integration test |
| **4** | Fix Issues 2+3 | Cross-triangle segments work | Cache implementation |
| **5** | Full optimization loop | Multi-iteration convergence | Production test |

---

## Phase 0: Verify Dependencies and Build Validation

### Objective
Ensure `topology_switcher.py` can be imported and all required methods/classes exist.

### Dependencies Checklist

**From `MeshTopology` (`src/core/mesh_topology.py`):**
- ✅ `get_triangles_sharing_edge(edge)` - Line 130
- ✅ `get_adjacent_edges_through_vertex(edge, vertex)` - Line 171

**From `TriMesh` (`src/core/tri_mesh.py`):**
- ✅ `get_triangle_edges(tri_idx)` - Line 147

**From `PartitionContour` (`src/core/contour_partition.py`):**
- ✅ `evaluate_variable_point(vp_idx)` - Line 340
- ✅ `variable_points` - List of VariablePoint
- ✅ `triangle_segments` - List of TriangleSegment
- ✅ `edge_to_varpoint` - Dict mapping edges to VP indices

**From `VariablePoint` (`src/core/contour_partition.py`):**
- ✅ `edge` - Tuple[int, int]
- ✅ `lambda_param` - float
- ✅ `evaluate(vertices)` - Returns 3D position

**From `SteinerHandler` (`src/core/steiner_handler.py`):**
- ✅ `TriplePoint.is_on_triangle_boundary(tol)` - Detection method
- ✅ `TriplePoint.steiner_point` - 3D position
- ✅ `TriplePoint.var_point_indices` - List of 3 VP indices
- ✅ `TriplePoint._point_to_segment_distance()` - Geometric utility

### Test 0: Import Validation
```python
# File: examples/test_topology_switcher_basic.py

from src.core.topology_switcher import TopologySwitcher
from src.core.mesh_topology import MeshTopology
from src.core.contour_partition import PartitionContour
from src.core.tri_mesh import TriMesh

# Load a simple mesh and partition
# Create MeshTopology
# Create TopologySwitcher
# Print status: "All imports successful"
```

**Expected Output:**
```
✓ All imports successful
✓ TopologySwitcher initialized
✓ MeshTopology initialized with N vertices, M edges, T triangles
```

**If this fails:** Fix import/syntax errors before proceeding.

---

## Phase 1: Test Both Type 1 & Type 2 Switching (Isolated)

### Objective
Test that **both types of topology switches** can execute successfully, without worrying about subsequent optimization or area calculations.

**Key insight:** Type 2 is implemented via Type 1 (it selects which VP to move based on triple point position, then calls `apply_type1_switch`). They should be tested together.

### What We're Testing
1. **Type 1 (Direct):** Can we detect a boundary VP and move it to an adjacent edge?
2. **Type 2 (Via Type 1):** Can we detect a boundary triple point, select the right VP, and migrate the triple point?
3. Are `vp.edge`, `vp.lambda_param`, and `edge_to_varpoint` correctly updated?
4. Does triple point re-identification work after a switch?

### What We're NOT Testing (Yet)
- ❌ Area calculation correctness (Issue 1 & 2 not fixed yet)
- ❌ Perimeter calculation correctness (Issue 1 not fixed yet)
- ❌ Optimization after switch
- ❌ Just testing the switching mechanics in isolation

### Test 1A: Detect Boundary VPs and Triple Points
```python
# File: examples/test_topology_switcher_basic.py

# 1. Load partition from a previous optimization
# 2. Manually set one VP's lambda to 0.05 (near boundary) to test Type 1
partition.variable_points[5].lambda_param = 0.05

# 3. Check Type 1 detection
boundary_vps = partition.get_boundary_variable_points(tol=0.1)
print(f"Detected {len(boundary_vps)} boundary VPs (Type 1)")
assert 5 in boundary_vps, "Failed to detect manually set boundary VP"

# 4. Check Type 2 detection (boundary triple points)
steiner_handler = SteinerHandler(mesh, partition)
boundary_tps = steiner_handler.get_boundary_triple_points(tol=0.1)
print(f"Detected {len(boundary_tps)} boundary triple points (Type 2)")

# Log details
for tp in boundary_tps:
    print(f"  Triple point in triangle {tp.triangle_idx}, VPs: {tp.var_point_indices}")
```

**Expected Output:**
```
✓ Detected 1 boundary VPs (Type 1)
✓ VP index 5 correctly identified
✓ Detected 2 boundary triple points (Type 2)
  Triple point in triangle 45, VPs: [12, 23, 34]
  Triple point in triangle 78, VPs: [34, 45, 56]
```

### Test 1B: Execute Single Type 1 Switch
```python
# File: examples/test_topology_switcher_basic.py

# 1. Create switcher
switcher = TopologySwitcher(mesh, partition, mesh_topology)

# 2. Before switch: record state
vp_idx = 5
vp_before = partition.variable_points[vp_idx]
old_edge = vp_before.edge
old_lambda = vp_before.lambda_param
print(f"Before: VP {vp_idx} on edge {old_edge}, λ={old_lambda:.3f}")

# 3. Apply switch
success = switcher.apply_type1_switch(vp_idx, tol=0.1)

# 4. After switch: verify state
vp_after = partition.variable_points[vp_idx]
new_edge = vp_after.edge
new_lambda = vp_after.lambda_param
print(f"After: VP {vp_idx} on edge {new_edge}, λ={new_lambda:.3f}")

# 5. Assertions
assert success, "Type 1 switch failed"
assert new_edge != old_edge, "VP did not move to new edge"
assert 0.05 < new_lambda < 0.95, "New lambda not in valid range"
assert old_edge not in partition.edge_to_varpoint, "Old edge still in map"
assert partition.edge_to_varpoint[new_edge] == vp_idx, "New edge not in map"
```

**Expected Output:**
```
Before: VP 5 on edge (12, 34), λ=0.050
  Candidate edges: [(11, 12), (12, 45)]
  Testing edge (11, 12): total_dist = 1.234
  Testing edge (12, 45): total_dist = 1.156
  Selected: (12, 45) with min distance
After: VP 5 on edge (12, 45), λ=0.100
✓ Type 1 switch successful
✓ edge_to_varpoint correctly updated
✓ VP position is valid
```

### Test 1C: Verify VP Position Consistency
```python
# File: examples/test_topology_switcher_basic.py

# Compute 3D position using new edge and lambda
pos_computed = partition.evaluate_variable_point(vp_idx)
v1, v2 = new_edge
p1 = mesh.vertices[v1]
p2 = mesh.vertices[v2]
pos_expected = new_lambda * p1 + (1 - new_lambda) * p2

distance = np.linalg.norm(pos_computed - pos_expected)
print(f"Position error: {distance:.2e}")
assert distance < 1e-10, "VP position mismatch"
```

**Expected Output:**
```
✓ VP position matches expected location
✓ Position error: 2.34e-12
```

### Test 1D: Execute Type 2 Switch (Via Type 1)
```python
# File: examples/test_topology_switcher_basic.py

# 1. Get boundary triple points from Test 1A
if len(boundary_tps) > 0:
    tp = boundary_tps[0]
    print(f"\nTesting Type 2 switch for triple point in triangle {tp.triangle_idx}")
    
    # 2. Select which VP to move
    vp_to_move = switcher.select_variable_point_for_type2(tp)
    print(f"  Selected VP {vp_to_move} for Type 2 switch")
    
    vp_before = partition.variable_points[vp_to_move]
    print(f"  VP on edge {vp_before.edge}, λ={vp_before.lambda_param:.3f}")
    
    # 3. Apply Type 1 switch (triggered by Type 2)
    success = switcher.apply_type1_switch(vp_to_move, tol=0.1)
    
    # 4. Verify VP moved
    vp_after = partition.variable_points[vp_to_move]
    print(f"  VP moved to edge {vp_after.edge}, λ={vp_after.lambda_param:.3f}")
    
    assert success, "Type 2 switch (via Type 1) failed"
    assert vp_after.edge != vp_before.edge, "VP did not move"
    
    print("✓ Type 2 switch successful")
else:
    print("⚠️ No boundary triple points detected, skipping Type 2 test")
```

**Expected Output:**
```
Testing Type 2 switch for triple point in triangle 45
  Selected VP 12 for Type 2 switch
  VP on edge (5, 8), λ=0.087
  Candidate edges: [(5, 9), (5, 11)]
  Testing edge (5, 9): total_dist = 1.234
  Testing edge (5, 11): total_dist = 1.156
  Selected: (5, 11) with min distance
  VP moved to edge (5, 11), λ=0.100
✓ Type 2 switch successful
```

### What to Check For
- ✅ Type 1: Switch completes without crashes
- ✅ Type 1: `vp.edge` changes, `vp.lambda_param` valid, `edge_to_varpoint` updated
- ✅ Type 2: Correct VP is selected (closest to vertex)
- ✅ Type 2: Switch executes via `apply_type1_switch()`
- ⚠️ **DO NOT** compute area or perimeter yet (Issues 1 & 2 not fixed)
- ⚠️ **DO NOT** check if triple point migrated yet (need to rebuild triangle_segments first)

### If This Fails
- **Type 1**: Debug candidate edge selection, check `_get_triangle_local_candidates()`
- **Type 1**: Check `_compute_total_segment_length()`
- **Type 2**: Debug `select_variable_point_for_type2()`, check VP selection logic
- **Type 2**: Check `_find_closest_edge_to_steiner()`, verify it finds correct edge
- Verify mesh topology connectivity

---

## Phase 2: Fix Issue 1 (Stale `triangle_segments`)

### Objective
Implement `rebuild_triangle_segments_from_current_vps()` so that area/perimeter calculations work after a switch.

### Implementation Steps

#### Step 2.1: Add Rebuild Method to `PartitionContour`
**File:** `src/core/contour_partition.py`

```python
def rebuild_triangle_segments_from_current_vps(self) -> None:
    """
    Rebuild triangle_segments from current variable_points state.
    
    Called after topology switches to ensure triangle_segments reflects
    the new VP positions on mesh edges.
    
    Algorithm:
    1. Clear self.triangle_segments
    2. Iterate through all mesh triangles
    3. For each triangle, check which current VPs are on its edges
    4. Group VPs by cell membership to determine segment type
    5. Create new TriangleSegment objects
    """
    self.logger.info("Rebuilding triangle_segments from current variable points...")
    
    # Clear old list
    self.triangle_segments.clear()
    
    # Create map: edge -> vp_idx for fast lookup
    edge_to_vp = {}
    for vp in self.variable_points:
        normalized_edge = tuple(sorted(vp.edge))
        edge_to_vp[normalized_edge] = vp.global_idx
    
    # Iterate through all triangles
    for tri_idx, face in enumerate(self.mesh.faces):
        v1, v2, v3 = face
        tri_edges = [
            tuple(sorted([v1, v2])),
            tuple(sorted([v2, v3])),
            tuple(sorted([v3, v1]))
        ]
        
        # Find which edges have VPs
        boundary_edges = []
        var_point_indices = []
        
        for edge in tri_edges:
            if edge in edge_to_vp:
                boundary_edges.append(edge)
                var_point_indices.append(edge_to_vp[edge])
        
        # Only create TriangleSegment if triangle has VPs on boundary
        if len(var_point_indices) >= 2:
            # Get vertex labels from indicator functions
            vertex_labels = tuple(np.argmax(self.indicator_functions[v]) for v in face)
            
            # Create TriangleSegment
            tri_seg = TriangleSegment(
                triangle_idx=tri_idx,
                vertex_indices=(v1, v2, v3),
                vertex_labels=vertex_labels,
                boundary_edges=boundary_edges,
                var_point_indices=var_point_indices
            )
            self.triangle_segments.append(tri_seg)
    
    # Log statistics
    num_two_cell = sum(1 for ts in self.triangle_segments if ts.num_cells() == 2)
    num_triple = sum(1 for ts in self.triangle_segments if ts.is_triple_point())
    self.logger.info(f"Rebuilt {len(self.triangle_segments)} triangle segments: "
                    f"{num_two_cell} two-cell, {num_triple} triple-point")
```

#### Step 2.2: Call Rebuild in `PerimeterOptimizer.reinitialize_after_switches()`
**File:** `src/core/perimeter_optimizer.py`

```python
def reinitialize_after_switches(self):
    """
    Reinitialize calculators after topology switches.
    
    Steps:
    1. Rebuild partition.triangle_segments (Issue 1 fix)
    2. Recreate AreaCalculator
    3. Recreate PerimeterCalculator
    4. Recreate SteinerHandler (recompute triple points)
    """
    self.logger.info("Reinitializing after topology switches...")
    
    # FIX ISSUE 1: Rebuild triangle_segments
    self.partition.rebuild_triangle_segments_from_current_vps()
    
    # Recreate calculators (they cache triangle_segments)
    self.area_calc = AreaCalculator(self.mesh, self.partition)
    self.perim_calc = PerimeterCalculator(self.mesh, self.partition)
    
    # Reidentify triple points
    self.partition.identify_triple_points()
    self.steiner_handler = SteinerHandler(self.mesh, self.partition)
    
    self.logger.info("Reinitialize complete")
```

### Test 2A: Verify Rebuild Correctness
```python
# File: examples/test_topology_switcher_basic.py

# 1. Count triangle_segments before switch
num_before = len(partition.triangle_segments)
print(f"Triangle segments before: {num_before}")

# 2. Apply Type 1 switch
switcher.apply_type1_switch(vp_idx=5, tol=0.1)

# 3. Rebuild triangle_segments
partition.rebuild_triangle_segments_from_current_vps()

# 4. Count after
num_after = len(partition.triangle_segments)
print(f"Triangle segments after: {num_after}")

# 5. Verify consistency
for tri_seg in partition.triangle_segments:
    for vp_idx in tri_seg.var_point_indices:
        vp = partition.variable_points[vp_idx]
        # Check VP is actually on one of the triangle's edges
        assert vp.edge in tri_seg.boundary_edges, \
            f"VP {vp_idx} on edge {vp.edge} not in tri_seg.boundary_edges"
```

**Expected Output:**
```
Triangle segments before: 234
  After Type 1 switch (VP 5: edge (12,34) → (12,45))
Triangle segments after: 234
✓ All VPs in triangle_segments are on correct edges
✓ No stale entries found
```

### Test 2B: Verify Area Calculation Works
```python
# File: examples/test_topology_switcher_basic.py

# 1. Compute areas after switch + rebuild
area_calc = AreaCalculator(mesh, partition)
areas = area_calc.compute_areas(partition.get_variable_vector())

print(f"Cell areas: {areas}")
print(f"Total area: {sum(areas):.6f}")
print(f"Mesh area: {mesh.get_total_area():.6f}")

# 2. Verify area conservation
assert abs(sum(areas) - mesh.get_total_area()) < 1e-6, \
    "Area not conserved after switch"
```

**Expected Output:**
```
Cell areas: [0.523, 0.477]  # Example for 2-cell partition
Total area: 1.000000
Mesh area: 1.000000
✓ Area conserved after Type 1 switch
```

### If This Fails
- Check `rebuild_triangle_segments_from_current_vps()` logic
- Verify `edge_to_vp` map is correct
- Check that all VPs are on actual mesh edges
- Print detailed debug info about which triangles changed

---

## Phase 3: Verify Switching + Area Calculations Work Together

### Objective
After fixing Issue 1, verify that **area and perimeter calculations work correctly after topology switches**.

**Key test:** Can we apply switches, rebuild `triangle_segments`, and still compute correct areas?

### What We're Testing
1. Does `rebuild_triangle_segments_from_current_vps()` correctly rebuild the list?
2. Do area calculations work after rebuild?
3. Is area conserved after switches?
4. Does triple point migration work correctly? (Type 2 verification)

### Test 3A: Verify Triple Point Migration (Type 2 Complete Test)
```python
# File: examples/test_topology_switcher_basic.py

# This is the completion of Test 1D - now we can verify triple point migration

# 1. Get boundary triple points (from Phase 1)
steiner_handler = SteinerHandler(mesh, partition)
boundary_tps = steiner_handler.get_boundary_triple_points(tol=0.1)

if len(boundary_tps) > 0:
    tp = boundary_tps[0]
    old_triangle = tp.triangle_idx
    vps_in_tp = set(tp.var_point_indices)
    
    print(f"Triple point in triangle {old_triangle}, VPs: {vps_in_tp}")
    
    # 2. Select and move VP
    vp_to_move = switcher.select_variable_point_for_type2(tp)
    success = switcher.apply_type1_switch(vp_to_move, tol=0.1)
    
    # 3. Rebuild triangle_segments (Issue 1 fix)
    partition.rebuild_triangle_segments_from_current_vps()
    
    # 4. Recompute triple points
    partition.identify_triple_points()
    steiner_handler_new = SteinerHandler(mesh, partition)
    
    # 5. Find migrated triple point
    found_migration = False
    for tp_new in steiner_handler_new.triple_points:
        if set(tp_new.var_point_indices) == vps_in_tp:
            new_triangle = tp_new.triangle_idx
            if new_triangle != old_triangle:
                print(f"✓ Triple point migrated: triangle {old_triangle} → {new_triangle}")
                found_migration = True
            break
    
    assert found_migration, "Triple point did not migrate to new triangle"
```

**Expected Output:**
```
Triple point in triangle 45, VPs: {12, 23, 34}
  Applying Type 2 switch...
  VP 12 moved from edge (5,8) to (5,11)
  Rebuilding triangle_segments...
  Rebuilt 234 triangle segments
✓ Triple point migrated: triangle 45 → triangle 78
```

### Test 3B: Verify Area Conservation After Switches
```python
# File: examples/test_topology_switcher_basic.py

# 1. Apply a Type 1 switch
print("\nTesting area calculation after Type 1 switch...")
vp_idx = boundary_vps[0]
switcher.apply_type1_switch(vp_idx, tol=0.1)

# 2. Rebuild triangle_segments
partition.rebuild_triangle_segments_from_current_vps()

# 3. Compute areas
area_calc = AreaCalculator(mesh, partition)
areas = area_calc.compute_areas(partition.get_variable_vector())

# 4. Verify area conservation
total_area = sum(areas)
mesh_area = mesh.get_total_area()
error = abs(total_area - mesh_area)

print(f"Cell areas: {areas}")
print(f"Total area: {total_area:.8f}")
print(f"Mesh area: {mesh_area:.8f}")
print(f"Error: {error:.2e}")

assert error < 1e-6, f"Area not conserved (error: {error})"
print("✓ Area conserved after Type 1 switch")
```

**Expected Output:**
```
Testing area calculation after Type 1 switch...
  VP 5 switched: edge (12,34) → (12,45)
  Rebuilding triangle_segments...
Cell areas: [0.523142, 0.476858]
Total area: 1.00000000
Mesh area: 1.00000000
Error: 2.3e-12
✓ Area conserved after Type 1 switch
```

### Test 3C: Verify Perimeter Calculation After Switches
```python
# File: examples/test_topology_switcher_basic.py

# 1. Compute perimeter after switch
perim_calc = PerimeterCalculator(mesh, partition)
perimeter = perim_calc.compute_total_perimeter(partition.get_variable_vector())

print(f"Total perimeter after switch: {perimeter:.6f}")

# 2. Verify it's a valid positive number
assert perimeter > 0, "Perimeter is not positive"
assert not np.isnan(perimeter), "Perimeter is NaN"
assert not np.isinf(perimeter), "Perimeter is infinite"

print("✓ Perimeter calculation works after switch")
```

**Expected Output:**
```
Total perimeter after switch: 3.456789
✓ Perimeter calculation works after switch
```

### What to Check For
- ✅ Type 2: Triple point migrates to adjacent triangle
- ✅ Area: Total area equals mesh area (conservation)
- ✅ Perimeter: Valid positive number
- ✅ `triangle_segments` correctly rebuilt
- ⚠️ **May still fail** if segments cross multiple triangles (Issue 2 not fixed yet)

### If This Fails
- **Triple point doesn't migrate**: Check `rebuild_triangle_segments_from_current_vps()` logic
- **Area not conserved**: Check if segments are crossing triangles (Issue 2)
- **Perimeter is NaN**: Check `PerimeterCalculator` segment extraction
- Debug: Print `triangle_segments` before/after to see changes

---

## Phase 4: Fix Issues 2+3 (Cross-Triangle Segments + Caching)

### Objective
Handle segments that cross multiple triangles after topology switching.

### Implementation Steps

#### Step 4.1: Add `SegmentCrossingInfo` Dataclass
**File:** `src/core/contour_partition.py` (near TriangleSegment definition)

```python
@dataclass
class SegmentCrossingInfo:
    """
    Precomputed geometric intersection for segment crossing triangle.
    
    Created during topology switching, used during area calculation.
    """
    segment: Tuple[int, int]        # (vp_i, vp_j)
    triangle_idx: int                # Triangle being crossed
    entry_point: np.ndarray          # 3D coords where segment enters
    exit_point: np.ndarray           # 3D coords where segment exits
    entry_edge: Tuple[int, int]      # Mesh edge crossed on entry
    exit_edge: Tuple[int, int]       # Mesh edge crossed on exit
    cell_idx: int                    # Which cell this segment belongs to
```

#### Step 4.2: Add Cache to `PartitionContour.__init__()`
**File:** `src/core/contour_partition.py`

```python
def __init__(self, mesh, indicator_functions, boundary_topology=None):
    # ... existing initialization ...
    
    # NEW: Cache for cross-triangle segment intersections (Issue 3)
    self.segment_crossing_cache: Dict[int, List[SegmentCrossingInfo]] = {}
    #   Key: triangle_idx
    #   Value: List of segments crossing this triangle (with precomputed intersections)
```

#### Step 4.3: Compute Cache in `TopologySwitcher._move_variable_point()`
**File:** `src/core/topology_switcher.py`

```python
def _move_variable_point(self, vp_idx, new_edge, new_lambda):
    """
    Move VP and compute segment crossing cache for affected triangles.
    """
    vp = self.partition.variable_points[vp_idx]
    old_edge = vp.edge
    
    # Update edge_to_varpoint (existing code)
    if old_edge in self.partition.edge_to_varpoint:
        del self.partition.edge_to_varpoint[old_edge]
    self.partition.edge_to_varpoint[new_edge] = vp_idx
    
    # Update variable point (existing code)
    vp.edge = new_edge
    vp.lambda_param = new_lambda
    
    # NEW: Compute segment crossing cache for this VP's segments
    self._update_segment_crossing_cache(vp_idx)

def _update_segment_crossing_cache(self, vp_idx: int) -> None:
    """
    Compute and cache geometric intersections for segments involving this VP.
    
    For each segment (vp_idx, neighbor):
    1. Find all triangles between the two edges
    2. Compute line-segment intersections with triangle boundaries
    3. Store in partition.segment_crossing_cache
    """
    # Get neighboring VPs
    neighbors = self._get_neighboring_variable_points(vp_idx)
    
    for neighbor_idx in neighbors:
        segment = tuple(sorted([vp_idx, neighbor_idx]))
        
        # Get positions
        pos_vp = self.partition.evaluate_variable_point(vp_idx)
        pos_neighbor = self.partition.evaluate_variable_point(neighbor_idx)
        
        # Get edges
        vp_edge = self.partition.variable_points[vp_idx].edge
        neighbor_edge = self.partition.variable_points[neighbor_idx].edge
        
        # If edges are on different triangles, compute crossing
        if not self._edges_share_triangle(vp_edge, neighbor_edge):
            crossing_info = self._compute_segment_crossing(
                segment, pos_vp, pos_neighbor, vp_edge, neighbor_edge
            )
            
            # Store in cache
            for tri_idx, info in crossing_info.items():
                if tri_idx not in self.partition.segment_crossing_cache:
                    self.partition.segment_crossing_cache[tri_idx] = []
                self.partition.segment_crossing_cache[tri_idx].append(info)

def _compute_segment_crossing(self, segment, pos1, pos2, edge1, edge2):
    """
    Compute geometric line-segment intersections with mesh triangles.
    
    Returns:
        Dict[tri_idx] -> SegmentCrossingInfo
    """
    # Implementation: geometric line-edge intersection
    # (This is the complex geometric computation that gets cached)
    pass
```

#### Step 4.4: Refactor `AreaCalculator` to Use Cache
**File:** `src/core/area_calculator.py`

```python
def _partial_area_two_inside(self, tri_idx, face, phi_vals, ...):
    """
    Compute area contribution with hybrid strategy:
    1. Check cache for precomputed crossings
    2. If not in cache, use original VP-on-edge logic
    """
    # NEW: Check cache first
    if tri_idx in self.partition.segment_crossing_cache:
        # Use precomputed intersections
        return self._partial_area_from_cache(tri_idx, face, phi_vals, ...)
    
    # Original logic (fast path for unchanged triangles)
    # ... existing code ...
```

### Test 4A: Verify Cache Population
```python
# File: examples/test_topology_switcher_basic.py

# 1. Check cache before switch
print(f"Cache entries before: {len(partition.segment_crossing_cache)}")

# 2. Apply Type 1 switch
switcher.apply_type1_switch(vp_idx=5, tol=0.1)

# 3. Check cache after
print(f"Cache entries after: {len(partition.segment_crossing_cache)}")

# 4. Inspect cache
for tri_idx, crossings in partition.segment_crossing_cache.items():
    print(f"Triangle {tri_idx}: {len(crossings)} crossings")
    for crossing in crossings:
        print(f"  Segment {crossing.segment}: {crossing.entry_edge} → {crossing.exit_edge}")
```

**Expected Output:**
```
Cache entries before: 0
Cache entries after: 3
Triangle 78: 1 crossings
  Segment (5, 12): (7,8) → (8,9)
Triangle 79: 1 crossings
  Segment (5, 23): (7,9) → (9,10)
✓ Cache populated for affected triangles
```

### Test 4B: Verify Area Calculation with Cache
```python
# File: examples/test_topology_switcher_basic.py

# 1. Compute areas using new cache-aware AreaCalculator
area_calc = AreaCalculator(mesh, partition)
areas = area_calc.compute_areas(partition.get_variable_vector())

# 2. Check area conservation
total_area = sum(areas)
mesh_area = mesh.get_total_area()
error = abs(total_area - mesh_area)

print(f"Total area: {total_area:.8f}")
print(f"Mesh area: {mesh_area:.8f}")
print(f"Error: {error:.2e}")

assert error < 1e-6, f"Area not conserved (error: {error})"
```

**Expected Output:**
```
Total area: 1.00000023
Mesh area: 1.00000000
Error: 2.3e-7
✓ Area conserved with cross-triangle segments
```

### Test 4C: Full Optimization Cycle
```python
# File: examples/test_topology_switcher_basic.py

# 1. Apply switch + rebuild
switcher.apply_type1_switch(vp_idx=5, tol=0.1)
partition.rebuild_triangle_segments_from_current_vps()

# 2. Reinitialize optimizer
optimizer = PerimeterOptimizer(mesh, partition, target_area=mesh.get_total_area()/n_cells)
optimizer.reinitialize_after_switches()

# 3. Run one optimization iteration
lambda_vec = partition.get_variable_vector()
result = optimizer.optimize(lambda_vec, max_iter=50)

print(f"Optimization converged: {result.success}")
print(f"Final perimeter: {result.fun:.6f}")
print(f"Constraint violation: {max(abs(result.constr_violation)) if hasattr(result, 'constr_violation') else 0:.2e}")
```

**Expected Output:**
```
Optimization converged: True
Final perimeter: 3.456789
Constraint violation: 2.3e-8
✓ Full optimization cycle works after topology switch
```

---

## Phase 5: Integration Testing

### Objective
Test complete topology switching loop with multiple iterations.

### Test 5: Multi-Iteration Topology Switching
```python
# File: examples/test_topology_switcher_integration.py

# Load initial relaxed solution
# Run full topology switching loop (as in refine_perimeter.py)

for iteration in range(max_topology_iterations):
    # 1. Optimize
    result = optimizer.optimize(lambda_vec, max_iter=200)
    
    # 2. Check for switches
    switch_info = optimizer.detect_topology_switches(tol=0.1)
    
    if not switch_info['type1_switches'] and not switch_info['type2_switches']:
        break
    
    # 3. Apply switches
    optimizer.apply_topology_switches(switch_info, switch_tol=0.1)
    
    # 4. Reinitialize
    optimizer.reinitialize_after_switches()
    
    # 5. Verify integrity
    areas = optimizer.area_calc.compute_areas(lambda_vec)
    assert abs(sum(areas) - mesh.get_total_area()) < 1e-6

print(f"Converged after {iteration+1} topology iterations")
```

**Expected Output:**
```
Iteration 0:
  Perimeter: 3.456789
  Type 1 switches: 2
  Type 2 switches: 1
  Applied 3 switches
Iteration 1:
  Perimeter: 3.423456
  Type 1 switches: 0
  Type 2 switches: 0
  No switches detected
✓ Converged after 2 topology iterations
✓ All area constraints satisfied
```

---

## Summary Table: What Gets Tested When

| Phase | Type 1 | Type 2 | Issue 1 Fixed | Issue 2+3 Fixed | Area/Perim Correct |
|---|---|---|---|---|---|
| **0** | - | - | ❌ | ❌ | - |
| **1** | ✅ Basic switch | ✅ Basic switch | ❌ | ❌ | ❌ Not checked |
| **2** | ✅ | ✅ | ✅ Fixed | ❌ | ❌ Not tested |
| **3** | ✅ | ✅ Migration verified | ✅ | ❌ | ✅ Tested (may fail) |
| **4** | ✅ | ✅ | ✅ | ✅ Fixed | ✅ Full |
| **5** | ✅ | ✅ | ✅ | ✅ | ✅ Multi-iter |

**Note:** In Phase 1, we test that both switch types execute (VPs move). In Phase 3, we verify the complete behavior (triple point migration, area conservation).

---

## File Structure for Tests

```
examples/
├── test_topology_switcher_basic.py         # Phases 0-4
├── test_topology_switcher_integration.py   # Phase 5
└── refine_perimeter.py                     # Production script (unchanged)
```

---

## Rollback Strategy

If any phase fails catastrophically:
1. **Phase 0-1 fails**: Fix imports/syntax in `topology_switcher.py`
2. **Phase 2 fails**: Isolate `rebuild_triangle_segments` logic, add debug prints
3. **Phase 3 fails**: Test Type 2 detection separately from Type 1 execution
4. **Phase 4 fails**: Implement cache without refactoring `AreaCalculator` first (manual override)
5. **Phase 5 fails**: Run phases 1-4 individually to isolate regression

---

## Time Estimates

| Phase | Estimated Time | Cumulative |
|---|---|---|
| Phase 0 | 30 min | 0.5h |
| Phase 1 | 2-3 hours | 3.5h |
| Phase 2 | 2-3 hours | 6.5h |
| Phase 3 | 1-2 hours | 8.5h |
| Phase 4 | 4-6 hours | 14.5h |
| Phase 5 | 2-3 hours | 17.5h |

**Total**: ~17-18 hours over 3-4 days

**Note:** Phase 1 is longer now because it tests both Type 1 and Type 2 together.

---

## Success Criteria

✅ **Phase 0**: All imports work, no syntax errors  
✅ **Phase 1**: Both Type 1 and Type 2 switches execute (VPs move, edge selection works)  
✅ **Phase 2**: `triangle_segments` rebuild works, no stale entries  
✅ **Phase 3**: Area/perimeter calculations work after switches, triple points migrate correctly  
✅ **Phase 4**: Cross-triangle segments handled correctly with cache  
✅ **Phase 5**: Full multi-iteration topology switching converges  

---

## Next Step

Start with **Phase 0**: Create `examples/test_topology_switcher_basic.py` and verify all dependencies exist.


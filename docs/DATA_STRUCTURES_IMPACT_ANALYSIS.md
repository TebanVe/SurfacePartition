# Data Structures: Impact Analysis for Topology Switching

**Date:** November 22, 2025  
**Purpose:** Identify all data structures that need modification and assess impact on existing code

---

## Summary

**Good News**: Only **2 data structures** need modification:
1. `PartitionContour` - add ONE new attribute (`segment_crossing_cache`)
2. `VariablePoint` - **NO structural changes** (already has all needed fields)

**No structural changes needed** for:
- ❌ `TriangleSegment` - structure unchanged (just rebuild the list)
- ❌ `VariablePoint` - structure unchanged (just update values)

---

## 1. PartitionContour (Add New Attribute)

### Location
`src/core/contour_partition.py` lines 114-155

### Current Structure
```python
class PartitionContour:
    mesh: TriMesh
    n_cells: int
    variable_points: List[VariablePoint]
    triangle_segments: List[TriangleSegment]        # ← Rebuilt, not modified
    indicator_functions: np.ndarray
    edge_to_varpoint: Dict[Tuple[int, int], int]   # ← Updated, not modified
    triple_points: Optional[List]
```

### Required Change
```python
class PartitionContour:
    # ... existing attributes ...
    
    # NEW attribute (Issue 3 - caching):
    segment_crossing_cache: Dict[int, List[SegmentCrossingInfo]]
    #   Key: triangle_idx
    #   Value: List of segments crossing this triangle (with intersection points)
```

### Impact Assessment

**Where `PartitionContour` is instantiated:**
1. `examples/refine_perimeter.py:129` - Main optimization script ✅
2. `examples/surface_visualization.py:35,164` - Visualization ✅
3. `examples/debug_archive/test_*.py` - Multiple debug scripts ✅

**Impact**: 
- ✅ **NO BREAKING CHANGES** - New attribute is optional, defaults to empty dict
- ✅ Existing code continues to work without modifications
- ✅ Only topology switching code uses the cache

**Where it's saved/loaded:**
- `save_refined_contours()` (lines 479-536): Does **not** serialize internal data structures
  - Only saves: λ values, metadata, evaluated segments (visualization)
  - Cache is **transient** (not saved to disk)
- ✅ **NO IMPACT** on HDF5 serialization

---

## 2. SegmentCrossingInfo (New Dataclass)

### Location
`src/core/contour_partition.py` (to be added near TriangleSegment definition)

### Structure (New)
```python
@dataclass
class SegmentCrossingInfo:
    """
    Precomputed information about a segment crossing a triangle.
    Created during topology switching, used during area calculation.
    """
    segment: Tuple[int, int]           # (vp_i, vp_j)
    triangle_idx: int                   # Triangle being crossed
    entry_point: np.ndarray             # 3D coords where segment enters
    exit_point: np.ndarray              # 3D coords where segment exits
    entry_edge: Tuple[int, int]         # Edge crossed on entry
    exit_edge: Tuple[int, int]          # Edge crossed on exit
    cell_idx: int                       # Which cell this belongs to
```

### Impact Assessment
- ✅ **NEW** dataclass, no existing code affected
- ✅ Only used by topology switching code
- ✅ Not serialized (transient data)

---

## 3. TriangleSegment (NO Structural Changes)

### Location
`src/core/contour_partition.py` lines 81-112

### Current Structure
```python
@dataclass
class TriangleSegment:
    triangle_idx: int
    vertex_indices: Tuple[int, int, int]
    vertex_labels: Tuple[int, int, int]
    boundary_edges: List[Tuple[int, int]]
    var_point_indices: List[int]
```

### Required Change
**NONE** - Structure remains **unchanged**!

**What changes**: The **list** `partition.triangle_segments` gets rebuilt, but each `TriangleSegment` object has the same structure.

### Impact Assessment

**Where `triangle_segments` is used:**

1. **`PerimeterOptimizer.diagnose_boundary_triple_points()`** (line 429)
   ```python
   for ts in self.partition.triangle_segments:
       if ts.triangle_idx == tp.triangle_idx and ts.is_triple_point():
   ```
   - ✅ **NO IMPACT** - Still iterates over list, structure unchanged

2. **`TopologySwitcher._get_neighboring_variable_points()`** (line 242)
   ```python
   for tri_seg in self.partition.triangle_segments:
       if vp_idx in tri_seg.var_point_indices:
   ```
   - ✅ **NO IMPACT** - Still iterates over list, structure unchanged

3. **`TopologySwitcher._find_closest_edge_to_steiner()`** (line 420)
   ```python
   for ts in self.partition.triangle_segments:
       if ts.triangle_idx == tri_idx and ts.is_triple_point():
   ```
   - ✅ **NO IMPACT** - Still iterates over list, structure unchanged

4. **`SteinerHandler.TriplePoint._compute_cell_to_varpoint_mapping()`** (line 104)
   ```python
   for ts in self.partition.triangle_segments:
       if ts.triangle_idx == self.triangle_idx and ts.is_triple_point():
   ```
   - ✅ **NO IMPACT** - Still iterates over list, structure unchanged

5. **`PartitionContour` methods** (multiple locations):
   - `_initialize_from_indicators()` - Creates new entries ✅
   - `_initialize_from_boundary_topology()` - Creates new entries ✅
   - `get_triangle_based_segments()` - Iterates over list ✅
   - `get_cell_segments_from_triangles()` - Iterates over list ✅
   - `to_visualization_format()` - Iterates over list ✅

**Critical insight**: 
- All existing code **reads** from `triangle_segments` list
- Only `_initialize_*` methods **write** to the list
- New `rebuild_triangle_segments_from_current_vps()` will be another **writer**
- ✅ **NO BREAKING CHANGES** to readers

---

## 4. VariablePoint (NO Structural Changes)

### Location
`src/core/contour_partition.py` lines 49-77

### Current Structure
```python
@dataclass
class VariablePoint:
    edge: Tuple[int, int]                  # ← Updated by topology switching
    lambda_param: float                    # ← Updated by topology switching
    global_idx: int                        # ← Unchanged
    belongs_to_cells: Set[int]             # ← Unchanged
```

### Required Change
**NONE** - Structure has all needed fields!

**What changes**: The **values** of `edge` and `lambda_param` get updated by `TopologySwitcher._move_variable_point()`.

### Impact Assessment

**Where `VariablePoint` fields are modified:**

1. **`TopologySwitcher._move_variable_point()`** (lines 309-335)
   ```python
   vp.edge = new_edge           # ← Updates existing field
   vp.lambda_param = new_lambda # ← Updates existing field
   ```
   - ✅ This is **already implemented** and working

2. **`PartitionContour.set_variable_vector()`** (existing method)
   ```python
   for i, vp in enumerate(self.variable_points):
       vp.lambda_param = lambda_vec[i]  # ← Updates existing field
   ```
   - ✅ Already handles λ updates during optimization

**Where `VariablePoint` is read:**
- `AreaCalculator._partial_area_*()` - Reads `lambda_param` ✅
- `PerimeterCalculator.compute_segment_length()` - Reads `lambda_param` ✅
- `SteinerHandler` - Reads `belongs_to_cells` ✅
- Topology switching detection - Reads `lambda_param` to check boundaries ✅

**Critical insight**:
- `VariablePoint` structure **already supports** topology switching!
- Only **values** change, not structure
- ✅ **NO BREAKING CHANGES**

---

## 5. edge_to_varpoint (Updated, Not Modified)

### Location
`src/core/contour_partition.py` line 154

### Current Structure
```python
edge_to_varpoint: Dict[Tuple[int, int], int]
# Key: (v1, v2) normalized edge tuple (smaller index first)
# Value: Variable point index on that edge
```

### Required Change
**NONE** - Structure unchanged!

**What changes**: Dictionary entries get **added/removed** when VPs move edges.

### Impact Assessment

**Where it's modified:**

1. **`TopologySwitcher._move_variable_point()`** (lines 328-331)
   ```python
   if old_edge in self.partition.edge_to_varpoint:
       del self.partition.edge_to_varpoint[old_edge]  # Remove old entry
   self.partition.edge_to_varpoint[new_edge] = vp_idx  # Add new entry
   ```
   - ✅ Standard dict operations, no structural change

2. **`PartitionContour._initialize_*` methods**
   ```python
   self.edge_to_varpoint[normalized_edge] = vp.global_idx
   ```
   - ✅ Creation/initialization, no change to structure

**Where it's read:**

1. **`AreaCalculator._partial_area_two_inside()`** (lines 283, 287-288)
   ```python
   if edge1 not in self.partition.edge_to_varpoint:
       return 0.0, ...  # ← This is why Issue 2 exists!
   vp_idx1 = self.partition.edge_to_varpoint[edge1]
   ```
   - ⚠️ **This is where the problem occurs** (Issue 2)
   - Solution: Cache + fallback strategy (Issue 3)

2. **`AreaCalculator._partial_area_one_inside()`** (lines 373, 376-377)
   - ⚠️ Same issue as above

3. **`PartitionContour._initialize_from_boundary_topology()`** (line 298)
   ```python
   var_point_indices.append(self.edge_to_varpoint[normalized_edge])
   ```
   - ✅ No impact (initialization only)

**Critical insight**:
- Dictionary structure unchanged
- `AreaCalculator` needs to **handle missing keys** (Issue 2 solution)
- Cache strategy (Issue 3) avoids the lookup problem
- ✅ **NO STRUCTURAL CHANGES**

---

## 6. Other Data Structures (Completely Unchanged)

### TriMesh
- **Location**: `src/core/tri_mesh.py`
- **Impact**: None
- **Why**: Topology switching doesn't modify mesh

### MeshTopology
- **Location**: `src/core/mesh_topology.py`
- **Impact**: None
- **Why**: Precomputed connectivity is constant

### SteinerHandler / TriplePoint
- **Location**: `src/core/steiner_handler.py`
- **Impact**: None to structure
- **Behavior**: Re-initialized after switches (fresh instance)

### AreaCalculator
- **Location**: `src/core/area_calculator.py`
- **Impact**: Method refactoring (not structure changes)
- **What changes**: New methods added, existing methods get cache lookup

### PerimeterCalculator
- **Location**: `src/core/perimeter_calculator.py`
- **Impact**: None
- **Why**: Uses rebuilt `triangle_segments`, structure unchanged

---

## Summary Table

| Data Structure | Structural Change? | Value Changes? | Where Used | Breaking Changes? |
|---|---|---|---|---|
| `PartitionContour` | ✅ Add 1 attribute | No | All modules | ❌ No (optional attr) |
| `SegmentCrossingInfo` | ✅ New dataclass | N/A | Topology switching | ❌ No (new code) |
| `TriangleSegment` | ❌ No | No | 5 modules | ❌ No (list rebuilt) |
| `VariablePoint` | ❌ No | ✅ Yes (edge, λ) | All modules | ❌ No (existing fields) |
| `edge_to_varpoint` | ❌ No | ✅ Yes (dict entries) | 2 modules | ❌ No (dict ops) |
| `TriMesh` | ❌ No | No | All modules | ❌ No |
| `MeshTopology` | ❌ No | No | Topology switching | ❌ No |
| `SteinerHandler` | ❌ No | No (rebuilt) | Optimization | ❌ No |
| `AreaCalculator` | ❌ No | No | Optimization | ❌ No (methods added) |
| `PerimeterCalculator` | ❌ No | No | Optimization | ❌ No |

---

## Risk Assessment

### Low Risk ✅
- **PartitionContour**: Adding one optional attribute
- **TriangleSegment**: No structural changes, list is rebuilt
- **VariablePoint**: No structural changes, already supports value updates
- **edge_to_varpoint**: Standard dict operations

### Medium Risk ⚠️
- **AreaCalculator refactoring**: Existing methods modified with cache logic
  - Mitigation: Hybrid approach (fast path for unchanged triangles)
  - Testing: Verify area calculations match before/after refactor

### No Risk ✅
- All other data structures unchanged

---

## Backwards Compatibility

**Will existing code break?**
- ❌ **NO** - All changes are additive or internal

**Can we run old code without topology switching?**
- ✅ **YES** - `segment_crossing_cache` defaults to empty dict
- ✅ **YES** - `AreaCalculator` hybrid approach uses fast path (unchanged behavior)

**Can we load old HDF5 files?**
- ✅ **YES** - Serialization unchanged (cache is transient)

---

## Conclusion

**Only 1 data structure requires modification:**
- `PartitionContour` - add `segment_crossing_cache` attribute

**All other changes are:**
- **Value updates** (not structure changes)
- **List rebuilding** (not structure changes)  
- **New code** (not modifications to existing structures)

**Risk**: ✅ **LOW** - Minimal structural changes, additive modifications, backwards compatible


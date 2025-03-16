use crate::bsp::Node;
use crate::float_types::{EPSILON, PI, Real, TAU};
use crate::plane::Plane;
use crate::polygon::Polygon;
use crate::vertex::Vertex;
use geo::{
    AffineOps, AffineTransform, BooleanOps, BoundingRect, Coord, CoordsIter, Geometry,
    GeometryCollection, LineString, MultiPolygon, Orient, Polygon as GeoPolygon, Rect,
    TriangulateEarcut, coord, line_string, orient::Direction,
};
use nalgebra::{
    Isometry3, Matrix3, Matrix4, Point3, Quaternion, Rotation3, Translation3,
    Unit, Vector3, partial_max, partial_min,
};

use crate::float_types::parry3d::{
    query::{Ray, RayCast},
    shape::{Shape, SharedShape, TriMesh, Triangle},
};
use crate::float_types::rapier3d::prelude::*;
use std::fmt::Debug;

#[cfg(feature = "hashmap")]
use hashbrown::HashMap;

#[cfg(feature = "chull-io")]
use chull::ConvexHullWrapper;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

#[cfg(feature = "offset")]
use geo_buf::{buffer_multi_polygon, buffer_polygon};

#[cfg(any(feature = "metaballs", feature = "sdf"))]
use fast_surface_nets::{SurfaceNetsBuffer, surface_nets};

#[cfg(feature = "metaballs")]
#[derive(Debug, Clone)]
pub struct MetaBall {
    pub center: Point3<Real>,
    pub radius: Real,
}

#[cfg(feature = "metaballs")]
impl MetaBall {
    pub const fn new(center: Point3<Real>, radius: Real) -> Self {
        Self { center, radius }
    }

    /// “Influence” function used by the scalar field for metaballs
    pub fn influence(&self, p: &Point3<Real>) -> Real {
        let dist_sq = (p - self.center).norm_squared() + EPSILON;
        self.radius * self.radius / dist_sq
    }
}

/// Summation of influences from multiple metaballs.
#[cfg(feature = "metaballs")]
fn scalar_field_metaballs(balls: &[MetaBall], p: &Point3<Real>) -> Real {
    let mut value = 0.0;
    for ball in balls {
        value += ball.influence(p);
    }
    value
}

/// The main CSG solid structure. Contains a list of 3D polygons, 2D polylines, and some metadata.
#[derive(Debug, Clone)]
pub struct CSG<S: Clone> {
    /// 3D polygons for volumetric shapes
    pub polygons: Vec<Polygon<S>>,

    /// 2D geometry
    pub geometry: GeometryCollection<Real>,

    /// Metadata
    pub metadata: Option<S>,
}

impl<S: Clone + Debug> CSG<S> where S: Clone + Send + Sync {
    /// Create an empty CSG
    pub fn new() -> Self {
        CSG {
            polygons: Vec::new(),
            geometry: GeometryCollection::default(),
            metadata: None,
        }
    }

    /// Helper to collect all vertices from the CSG.
    #[cfg(not(feature = "parallel"))]
    pub fn vertices(&self) -> Vec<Vertex> {
        self.polygons
            .iter()
            .flat_map(|p| p.vertices.clone())
            .collect()
    }

    /// Parallel helper to collect all vertices from the CSG.
    #[cfg(feature = "parallel")]
    pub fn vertices(&self) -> Vec<Vertex> {
        self.polygons
            .par_iter()
            .flat_map(|p| p.vertices.clone())
            .collect()
    }

    /// Build a CSG from an existing polygon list
    pub fn from_polygons(polygons: &[Polygon<S>]) -> Self {
        CSG {
            polygons: polygons.to_vec(),
            geometry: GeometryCollection::default(),
            metadata: None,
        }
    }

    /// Convert internal polylines into polygons and return along with any existing internal polygons.
    pub fn to_polygons(&self) -> Vec<Polygon<S>> {
        let mut all_polygons = Vec::new();

        for geom in &self.geometry {
            if let Geometry::Polygon(poly2d) = geom {
                // 1. Convert the outer ring to 3D.
                let mut outer_vertices_3d = Vec::new();
                for c in poly2d.exterior().coords_iter() {
                    outer_vertices_3d.push(
                        Vertex::new(Point3::new(c.x, c.y, 0.0), Vector3::z())
                    );
                }

                // Push as a new Polygon<S> if it has at least 3 vertices.
                if outer_vertices_3d.len() >= 3 {
                    all_polygons.push(Polygon::new(outer_vertices_3d, self.metadata.clone()));
                }

                // 2. Convert each interior ring (hole) into its own Polygon<S>.
                for ring in poly2d.interiors() {
                    let mut hole_vertices_3d = Vec::new();
                    for c in ring.coords_iter() {
                        hole_vertices_3d.push(
                            Vertex::new(Point3::new(c.x, c.y, 0.0), Vector3::z())
                        );
                    }

                    if hole_vertices_3d.len() >= 3 {
                        // If your `Polygon<S>` type can represent holes internally,
                        // adjust this to store hole_vertices_3d as a hole rather
                        // than a new standalone polygon.
                        all_polygons.push(Polygon::new(hole_vertices_3d, self.metadata.clone()));
                    }
                }
            }
            // else if let Geometry::LineString(ls) = geom {
            //     // Example of how you might convert a linestring to a polygon,
            //     // if desired. Omitted for brevity.
            // }
        }

        all_polygons
    }

    /// Create a CSG that holds *only* 2D geometry in a `geo::GeometryCollection`.
    pub fn from_geo(geometry: GeometryCollection<Real>, metadata: Option<S>) -> Self {
        let mut csg = CSG::new();
        csg.geometry = geometry;
        csg.metadata = metadata;
        csg
    }

    pub fn tessellate_2d(outer: &[[Real; 2]], holes: &[&[[Real; 2]]]) -> Vec<[Point3<Real>; 3]> {
        // Convert the outer ring into a `LineString`
        let outer_coords: Vec<Coord<Real>> = outer.iter().map(|&[x, y]| Coord { x, y }).collect();

        // Convert each hole into its own `LineString`
        let holes_coords: Vec<LineString<Real>> = holes
            .iter()
            .map(|hole| {
                let coords: Vec<Coord<Real>> = hole.iter().map(|&[x, y]| Coord { x, y }).collect();
                LineString::new(coords)
            })
            .collect();

        // Ear-cut triangulation on the polygon (outer + holes)
        let polygon = GeoPolygon::new(LineString::new(outer_coords), holes_coords);
        let triangulation = polygon.earcut_triangles_raw();
        let triangle_indices = triangulation.triangle_indices;
        let vertices = triangulation.vertices;

        // Convert the 2D result (x,y) into 3D triangles with z=0
        let mut result = Vec::with_capacity(triangle_indices.len() / 3);
        for tri in triangle_indices.chunks_exact(3) {
            let pts = [
                Point3::new(vertices[2 * tri[0]], vertices[2 * tri[0] + 1], 0.0),
                Point3::new(vertices[2 * tri[1]], vertices[2 * tri[1] + 1], 0.0),
                Point3::new(vertices[2 * tri[2]], vertices[2 * tri[2] + 1], 0.0),
            ];
            result.push(pts);
        }
        result
    }

    /// Return a new CSG representing union of the two CSG's.
    ///
    /// ```no_run
    /// let c = a.union(b);
    ///     +-------+            +-------+
    ///     |       |            |       |
    ///     |   a   |            |   c   |
    ///     |    +--+----+   =   |       +----+
    ///     +----+--+    |       +----+       |
    ///          |   b   |            |   c   |
    ///          |       |            |       |
    ///          +-------+            +-------+
    /// ```
    #[must_use = "Use new CSG representing space in both CSG's"]
    pub fn union(&self, other: &CSG<S>) -> CSG<S> {
        let mut a = Node::new(&self.polygons);
        let mut b = Node::new(&other.polygons);

        a.clip_to(&b);
        b.clip_to(&a);
        b.invert();
        b.clip_to(&a);
        b.invert();
        a.build(&b.all_polygons());

        // Extract polygons from geometry
        let polys1 = gc_to_polygons(&self.geometry);
        let polys2 = gc_to_polygons(&other.geometry);

        // Perform union on those polygons
        let unioned = polys1.union(&polys2); // This is valid if each is a MultiPolygon
        let oriented = unioned.orient(Direction::Default);

        // Wrap the unioned polygons + lines/points back into one GeometryCollection
        let mut final_gc = GeometryCollection::default();
        final_gc.0.push(Geometry::MultiPolygon(oriented));

        // re-insert lines & points from both sets:
        for g in &self.geometry.0 {
            match g {
                Geometry::Polygon(_) | Geometry::MultiPolygon(_) => {
                    // skip polygons
                }
                _ => final_gc.0.push(g.clone()),
            }
        }
        for g in &other.geometry.0 {
            match g {
                Geometry::Polygon(_) | Geometry::MultiPolygon(_) => {
                    // skip polygons
                }
                _ => final_gc.0.push(g.clone()),
            }
        }

        CSG {
            polygons: a.all_polygons(),
            geometry: final_gc,
            metadata: self.metadata.clone(),
        }
    }

    /// Return a new CSG representing diffarence of the two CSG's.
    ///
    /// ```no_run
    /// let c = a.difference(b);
    ///     +-------+            +-------+
    ///     |       |            |       |
    ///     |   a   |            |   c   |
    ///     |    +--+----+   =   |    +--+
    ///     +----+--+    |       +----+
    ///          |   b   |
    ///          |       |
    ///          +-------+
    /// ```
    #[must_use = "Use new CSG"]
    pub fn difference(&self, other: &CSG<S>) -> CSG<S> {
        let mut a = Node::new(&self.polygons);
        let mut b = Node::new(&other.polygons);

        // -- 3D --
        a.invert();
        a.clip_to(&b);
        b.clip_to(&a);
        b.invert();
        b.clip_to(&a);
        b.invert();
        a.build(&b.all_polygons());
        a.invert();

        // -- 2D geometry-based approach --
        let polys1 = gc_to_polygons(&self.geometry);
        let polys2 = gc_to_polygons(&other.geometry);

        // Perform difference on those polygons
        let differenced = polys1.difference(&polys2);
        let oriented = differenced.orient(Direction::Default);

        // Wrap the differenced polygons + lines/points back into one GeometryCollection
        let mut final_gc = GeometryCollection::default();
        final_gc.0.push(Geometry::MultiPolygon(oriented));

        // Re-insert lines & points from self only
        // (If you need to exclude lines/points that lie inside other, you'd need more checks here.)
        for g in &self.geometry.0 {
            match g {
                Geometry::Polygon(_) | Geometry::MultiPolygon(_) => {} // skip
                _ => final_gc.0.push(g.clone()),
            }
        }

        CSG {
            polygons: a.all_polygons(),
            geometry: final_gc,
            metadata: self.metadata.clone(),
        }
    }

    /// Return a new CSG representing intersection of the two CSG's.
    ///
    /// ```no_run
    /// let c = a.intersect(b);
    ///     +-------+
    ///     |       |
    ///     |   a   |
    ///     |    +--+----+   =   +--+
    ///     +----+--+    |       +--+
    ///          |   b   |
    ///          |       |
    ///          +-------+
    /// ```
    #[must_use = "Use new CSG"]
    pub fn intersection(&self, other: &CSG<S>) -> CSG<S> {
        let mut a = Node::new(&self.polygons);
        let mut b = Node::new(&other.polygons);

        // -- 3D --
        a.invert();
        b.clip_to(&a);
        b.invert();
        a.clip_to(&b);
        b.clip_to(&a);
        a.build(&b.all_polygons());
        a.invert();

        // -- 2D geometry-based approach --
        let polys1 = gc_to_polygons(&self.geometry);
        let polys2 = gc_to_polygons(&other.geometry);

        // Perform intersection on those polygons
        let intersected = polys1.intersection(&polys2);
        let oriented = intersected.orient(Direction::Default);

        // Wrap the intersected polygons + lines/points into one GeometryCollection
        let mut final_gc = GeometryCollection::default();
        final_gc.0.push(Geometry::MultiPolygon(oriented));

        // For lines and points: keep them only if they intersect in both sets
        // todo: detect intersection of non-polygons
        for g in &self.geometry.0 {
            match g {
                Geometry::Polygon(_) | Geometry::MultiPolygon(_) => {} // skip
                _ => final_gc.0.push(g.clone()),
            }
        }
        for g in &other.geometry.0 {
            match g {
                Geometry::Polygon(_) | Geometry::MultiPolygon(_) => {} // skip
                _ => final_gc.0.push(g.clone()),
            }
        }

        CSG {
            polygons: a.all_polygons(),
            geometry: final_gc,
            metadata: self.metadata.clone(),
        }
    }

    /// Return a new CSG representing space in this CSG excluding the space in the
    /// other CSG plus the space in the other CSG excluding the space in this CSG.
    ///
    /// ```no_run
    /// let c = a.xor(b);
    ///     +-------+            +-------+
    ///     |       |            |       |
    ///     |   a   |            |   a   |
    ///     |    +--+----+   =   |    +--+----+
    ///     +----+--+    |       +----+--+    |
    ///          |   b   |            |       |
    ///          |       |            |       |
    ///          +-------+            +-------+
    /// ```
    #[must_use = "Use new CSG"]
    pub fn xor(&self, other: &CSG<S>) -> CSG<S> {
        // A \ B
        let a_sub_b = self.difference(other);

        // B \ A
        let b_sub_a = other.difference(self);

        // Union those two
        a_sub_b.union(&b_sub_a)

        /* here in case 2D xor misbehaves as an alternate implementation
        // -- 2D geometry-based approach only (no polygon-based Node usage here) --
        let polys1 = gc_to_polygons(&self.geometry);
        let polys2 = gc_to_polygons(&other.geometry);

        // Perform symmetric difference (XOR)
        let xored = polys1.xor(&polys2);
        let oriented = xored.orient(Direction::Default);

        // Wrap in a new GeometryCollection
        let mut final_gc = GeometryCollection::default();
        final_gc.0.push(Geometry::MultiPolygon(oriented));

        // Re-insert lines & points from both sets
        for g in &self.geometry.0 {
            match g {
                Geometry::Polygon(_) | Geometry::MultiPolygon(_) => {}, // skip
                _ => final_gc.0.push(g.clone()),
            }
        }
        for g in &other.geometry.0 {
            match g {
                Geometry::Polygon(_) | Geometry::MultiPolygon(_) => {}, // skip
                _ => final_gc.0.push(g.clone()),
            }
        }

        CSG {
            // If you also want a polygon-based Node XOR, you'd need to implement that similarly
            polygons: self.polygons.clone(),
            geometry: final_gc,
            metadata: self.metadata.clone(),
        }
        */
    }

    /// Invert this CSG (flip inside vs. outside)
    pub fn inverse(&self) -> CSG<S> {
        let mut csg = self.clone();
        for p in &mut csg.polygons {
            p.flip();
        }
        csg
    }

    /// Creates a 2D square in the XY plane.
    ///
    /// # Parameters
    ///
    /// - `width`: the width of the square
    /// - `length`: the height of the square
    /// - `metadata`: optional metadata
    ///
    /// # Example
    /// ```no_run
    /// let sq2 = CSG::square(2.0, 3.0, None);
    /// ```
    pub fn square(width: Real, length: Real, metadata: Option<S>) -> Self {
        // In geo, a Polygon is basically (outer: LineString, Vec<LineString> for holes).
        let outer = line_string![
            (x: 0.0,   y: 0.0),
            (x: width, y: 0.0),
            (x: width, y: length),
            (x: 0.0,   y: length),
            (x: 0.0,   y: 0.0), // close explicitly
        ];
        let polygon_2d = GeoPolygon::new(outer, vec![]);

        CSG::from_geo(
            GeometryCollection(vec![Geometry::Polygon(polygon_2d)]),
            metadata,
        )
    }

    /// Creates a 2D circle in the XY plane.
    pub fn circle(radius: Real, segments: usize, metadata: Option<S>) -> Self {
        if segments < 3 {
            return CSG::new();
        }
        let mut coords = Vec::with_capacity(segments + 1);
        for i in 0..segments {
            let theta = 2.0 * PI * (i as Real) / (segments as Real);
            let x = radius * theta.cos();
            let y = radius * theta.sin();
            coords.push((x, y));
        }
        // close it
        coords.push((coords[0].0, coords[0].1));
        let polygon_2d = GeoPolygon::new(LineString::from(coords), vec![]);

        CSG::from_geo(
            GeometryCollection(vec![Geometry::Polygon(polygon_2d)]),
            metadata,
        )
    }

    /// Create a 2D metaball iso-contour in XY plane from a set of 2D metaballs.
    /// - `balls`: array of (center, radius).
    /// - `resolution`: (nx, ny) grid resolution for marching squares.
    /// - `iso_value`: threshold for the iso-surface.
    /// - `padding`: extra boundary beyond each ball's radius.
    /// - `metadata`: optional user metadata.
    pub fn metaball_2d(
        balls: &[(nalgebra::Point2<Real>, Real)],
        resolution: (usize, usize),
        iso_value: Real,
        padding: Real,
        metadata: Option<S>,
    ) -> CSG<S> {
        let (nx, ny) = resolution;
        if balls.is_empty() || nx < 2 || ny < 2 {
            return CSG::new();
        }

        // 1) Compute bounding box around all metaballs
        let mut min_x = Real::MAX;
        let mut min_y = Real::MAX;
        let mut max_x = -Real::MAX;
        let mut max_y = -Real::MAX;
        for (center, r) in balls {
            let rr = *r + padding;
            if center.x - rr < min_x { min_x = center.x - rr; }
            if center.x + rr > max_x { max_x = center.x + rr; }
            if center.y - rr < min_y { min_y = center.y - rr; }
            if center.y + rr > max_y { max_y = center.y + rr; }
        }

        let dx = (max_x - min_x) / (nx as Real - 1.0);
        let dy = (max_y - min_y) / (ny as Real - 1.0);

        // 2) Fill a grid with the summed “influence” minus iso_value
        fn scalar_field(balls: &[(nalgebra::Point2<Real>, Real)], x: Real, y: Real) -> Real {
            let mut v = 0.0;
            for (c, r) in balls {
                let dx = x - c.x;
                let dy = y - c.y;
                let dist_sq = dx * dx + dy * dy + EPSILON;
                v += (r * r) / dist_sq;
            }
            v
        }

        let mut grid = vec![0.0; nx * ny];
        let index = |ix: usize, iy: usize| -> usize { iy * nx + ix };
        for iy in 0..ny {
            let yv = min_y + (iy as Real) * dy;
            for ix in 0..nx {
                let xv = min_x + (ix as Real) * dx;
                let val = scalar_field(balls, xv, yv) - iso_value;
                grid[index(ix, iy)] = val;
            }
        }

        // 3) Marching squares -> line segments
        let mut contours = Vec::<LineString<Real>>::new();

        // Interpolator:
        let interpolate =
            |(x1, y1, v1): (Real, Real, Real), (x2, y2, v2): (Real, Real, Real)| -> (Real, Real) {
                let denom = (v2 - v1).abs();
                if denom < EPSILON {
                    (x1, y1)
                } else {
                    let t = -v1 / (v2 - v1); // crossing at 0
                    (x1 + t * (x2 - x1), y1 + t * (y2 - y1))
                }
            };

        for iy in 0..(ny - 1) {
            let y0 = min_y + (iy as Real) * dy;
            let y1 = min_y + ((iy + 1) as Real) * dy;

            for ix in 0..(nx - 1) {
                let x0 = min_x + (ix as Real)*dx;
                let x1 = min_x + ((ix+1) as Real)*dx;
    
                let v0 = grid[index(ix,   iy  )];
                let v1 = grid[index(ix+1, iy  )];
                let v2 = grid[index(ix+1, iy+1)];
                let v3 = grid[index(ix,   iy+1)];
    
                // classification
                let mut c = 0u8;
                if v0 >= 0.0 { c |= 1; }
                if v1 >= 0.0 { c |= 2; }
                if v2 >= 0.0 { c |= 4; }
                if v3 >= 0.0 { c |= 8; }
                if c == 0 || c == 15 {
                    continue; // no crossing
                }
    
                let corners = [
                    (x0, y0, v0),
                    (x1, y0, v1),
                    (x1, y1, v2),
                    (x0, y1, v3),
                ];
    
                let mut pts = Vec::new();
                // function to check each edge
                let mut check_edge = |mask_a: u8, mask_b: u8, a: usize, b: usize| {
                    let inside_a = (c & mask_a) != 0;
                    let inside_b = (c & mask_b) != 0;
                    if inside_a != inside_b {
                        let (px, py) = interpolate(corners[a], corners[b]);
                        pts.push((px, py));
                    }
                };

                check_edge(1, 2, 0, 1);
                check_edge(2, 4, 1, 2);
                check_edge(4, 8, 2, 3);
                check_edge(8, 1, 3, 0);

                // we might get 2 intersection points => single line segment
                // or 4 => two line segments, etc.
                // For simplicity, we just store them in a small open polyline:
                if pts.len() >= 2 {
                    let mut pl = LineString::new(vec![]);
                    for &(px, py) in &pts {
                        pl.0.push(coord! {x: px, y: py});
                    }
                    // Do not close. These are just line segments from this cell.
                    contours.push(pl);
                }
            }
        }

        // 4) Convert these line segments into geo::LineStrings or geo::Polygons if closed.
        //    We store them in a GeometryCollection.
        let mut gc = GeometryCollection::default();

        // If you want to unify them into continuous lines, you can do so,
        // but for now let's just push each as a separate line or polygon if closed.
        for pl in contours {
            let n = pl.coords_count();
            if n < 2 { continue; }

            // gather coords
            let coords: Vec<_> = (0..n)
                .map(|i| {
                    let v = pl.0[i];
                    (v.x, v.y)
                })
                .collect();

            // Check if first == last => closed
            let closed = {
                let first = coords[0];
                let last = coords[n - 1];
                let dx = first.0 - last.0;
                let dy = first.1 - last.1;
                (dx * dx + dy * dy).sqrt() < EPSILON
            };

            if closed {
                // Turn it into a Polygon
                let polygon_2d = GeoPolygon::new(LineString::from(coords.clone()), vec![]);
                gc.0.push(Geometry::Polygon(polygon_2d));
            } else {
                // It's an open line
                gc.0.push(Geometry::LineString(LineString::from(coords)));
            }
        }

        CSG::from_geo(gc, metadata)
    }

    /// Create a right prism (a box) that spans from (0, 0, 0)
    /// to (width, length, height). All dimensions must be >= 0.
    #[cfg(test)]
    pub(crate) fn cube(width: Real, length: Real, height: Real, metadata: Option<S>) -> CSG<S> {
        // Define the eight corner points of the prism.
        //    (x, y, z)
        let p000 = Point3::new(0.0, 0.0, 0.0);
        let p100 = Point3::new(width, 0.0, 0.0);
        let p110 = Point3::new(width, length, 0.0);
        let p010 = Point3::new(0.0, length, 0.0);

        let p001 = Point3::new(0.0, 0.0, height);
        let p101 = Point3::new(width, 0.0, height);
        let p111 = Point3::new(width, length, height);
        let p011 = Point3::new(0.0, length, height);

        // We’ll define 6 faces (each a Polygon), in an order that keeps outward-facing normals
        // and consistent (counter-clockwise) vertex winding as viewed from outside the prism.

        // Bottom face (z=0, normal approx. -Z)
        // p000 -> p100 -> p110 -> p010
        let bottom_normal = -Vector3::z();
        let bottom = Polygon::new(
            vec![
                Vertex::new(p000, bottom_normal),
                Vertex::new(p010, bottom_normal),
                Vertex::new(p110, bottom_normal),
                Vertex::new(p100, bottom_normal),
            ],
            metadata.clone(),
        );

        // Top face (z=depth, normal approx. +Z)
        // p001 -> p011 -> p111 -> p101
        let top_normal = Vector3::z();
        let top = Polygon::new(
            vec![
                Vertex::new(p001, top_normal),
                Vertex::new(p101, top_normal),
                Vertex::new(p111, top_normal),
                Vertex::new(p011, top_normal),
            ],
            metadata.clone(),
        );

        // Front face (y=0, normal approx. -Y)
        // p000 -> p001 -> p101 -> p100
        let front_normal = -Vector3::y();
        let front = Polygon::new(
            vec![
                Vertex::new(p000, front_normal),
                Vertex::new(p100, front_normal),
                Vertex::new(p101, front_normal),
                Vertex::new(p001, front_normal),
            ],
            metadata.clone(),
        );

        // Back face (y=height, normal approx. +Y)
        // p010 -> p110 -> p111 -> p011
        let back_normal = Vector3::y();
        let back = Polygon::new(
            vec![
                Vertex::new(p010, back_normal),
                Vertex::new(p011, back_normal),
                Vertex::new(p111, back_normal),
                Vertex::new(p110, back_normal),
            ],
            metadata.clone(),
        );

        // Left face (x=0, normal approx. -X)
        // p000 -> p010 -> p011 -> p001
        let left_normal = -Vector3::x();
        let left = Polygon::new(
            vec![
                Vertex::new(p000, left_normal),
                Vertex::new(p001, left_normal),
                Vertex::new(p011, left_normal),
                Vertex::new(p010, left_normal),
            ],
            metadata.clone(),
        );

        // Right face (x=width, normal approx. +X)
        // p100 -> p101 -> p111 -> p110
        let right_normal = Vector3::x();
        let right = Polygon::new(
            vec![
                Vertex::new(p100, right_normal),
                Vertex::new(p110, right_normal),
                Vertex::new(p111, right_normal),
                Vertex::new(p101, right_normal),
            ],
            metadata.clone(),
        );

        // Combine all faces into a CSG
        CSG::from_polygons(&[bottom, top, front, back, left, right])
    }

    /// Construct a sphere with radius, segments, stacks
    #[cfg(test)]
    pub(crate) fn sphere(
        radius: Real,
        segments: usize,
        stacks: usize,
        metadata: Option<S>,
    ) -> CSG<S> {
        let mut polygons = Vec::new();

        for i in 0..segments {
            for j in 0..stacks {
                let mut vertices = Vec::new();

                let vertex = |theta: Real, phi: Real| {
                    let dir =
                        Vector3::new(theta.cos() * phi.sin(), phi.cos(), theta.sin() * phi.sin());
                    Vertex::new(
                        Point3::new(dir.x * radius, dir.y * radius, dir.z * radius),
                        dir,
                    )
                };

                let t0 = i as Real / segments as Real;
                let t1 = (i + 1) as Real / segments as Real;
                let p0 = j as Real / stacks as Real;
                let p1 = (j + 1) as Real / stacks as Real;

                let theta0 = t0 * TAU;
                let theta1 = t1 * TAU;
                let phi0 = p0 * PI;
                let phi1 = p1 * PI;

                vertices.push(vertex(theta0, phi0));
                if j > 0 {
                    vertices.push(vertex(theta1, phi0));
                }
                if j < stacks - 1 {
                    vertices.push(vertex(theta1, phi1));
                }
                vertices.push(vertex(theta0, phi1));

                polygons.push(Polygon::new(vertices, metadata.clone()));
            }
        }
        CSG::from_polygons(&polygons)
    }

    /// Constructs a frustum between `start` and `end` with bottom radius = `radius1` and
    /// top radius = `radius2`. In the normal case, it creates side quads and cap triangles.
    /// However, if one of the radii is 0 (within EPSILON), then the degenerate face is treated
    /// as a single point and the side is stitched using triangles.
    ///
    /// # Parameters
    /// - `start`: the center of the bottom face
    /// - `end`: the center of the top face
    /// - `radius1`: the radius at the bottom face
    /// - `radius2`: the radius at the top face
    /// - `segments`: number of segments around the circle (must be ≥ 3)
    /// - `metadata`: optional metadata
    ///
    /// # Example
    /// ```
    /// let bottom = Point3::new(0.0, 0.0, 0.0);
    /// let top = Point3::new(0.0, 0.0, 5.0);
    /// // This will create a cone (bottom degenerate) because radius1 is 0:
    /// let cone = CSG::frustrum_ptp_special(bottom, top, 0.0, 2.0, 32, None);
    /// ```
    pub fn frustrum_ptp(
        start: Point3<Real>,
        end: Point3<Real>,
        radius1: Real,
        radius2: Real,
        segments: usize,
        metadata: Option<S>,
    ) -> CSG<S> {
        // Compute the axis and check that start and end do not coincide.
        let s = start.coords;
        let e = end.coords;
        let ray = e - s;
        if ray.norm_squared() < EPSILON {
            return CSG::new();
        }
        let axis_z = ray.normalize();
        // Pick an axis not parallel to axis_z.
        let axis_x = if axis_z.y.abs() > 0.5 {
            Vector3::x()
        } else {
            Vector3::y()
        }
        .cross(&axis_z)
        .normalize();
        let axis_y = axis_x.cross(&axis_z).normalize();
        // The cap centers for the bottom and top.
        let start_v = Vertex::new(start, -axis_z);
        let end_v = Vertex::new(end, axis_z);

        // A closure that returns a vertex on the lateral surface.
        // For a given stack (0.0 for bottom, 1.0 for top), slice (fraction along the circle),
        // and a normal blend factor (used for cap smoothing), compute the vertex.
        let point = |stack: Real, slice: Real, normal_blend: Real| {
            // Linear interpolation of radius.
            let r = radius1 * (1.0 - stack) + radius2 * stack;
            let angle = slice * TAU;
            let radial_dir = axis_x * angle.cos() + axis_y * angle.sin();
            let pos = s + ray * stack + radial_dir * r;
            let normal = radial_dir * (1.0 - normal_blend.abs()) + axis_z * normal_blend;
            Vertex::new(Point3::from(pos), normal.normalize())
        };

        let mut polygons = Vec::new();

        // Special-case flags for degenerate faces.
        let bottom_degenerate = radius1.abs() < EPSILON;
        let top_degenerate = radius2.abs() < EPSILON;

        // If both faces are degenerate, we cannot build a meaningful volume.
        if bottom_degenerate && top_degenerate {
            return CSG::new();
        }

        // For each slice of the circle (0..segments)
        for i in 0..segments {
            let slice0 = i as Real / segments as Real;
            let slice1 = (i + 1) as Real / segments as Real;

            // In the normal frustrum_ptp, we always add a bottom cap triangle (fan) and a top cap triangle.
            // Here, we only add the cap triangle if the corresponding radius is not degenerate.
            if !bottom_degenerate {
                // Bottom cap: a triangle fan from the bottom center to two consecutive points on the bottom ring.
                polygons.push(Polygon::new(
                    vec![
                        start_v.clone(),
                        point(0.0, slice0, -1.0),
                        point(0.0, slice1, -1.0),
                    ],
                    metadata.clone(),
                ));
            }
            if !top_degenerate {
                // Top cap: a triangle fan from the top center to two consecutive points on the top ring.
                polygons.push(Polygon::new(
                    vec![
                        end_v.clone(),
                        point(1.0, slice1, 1.0),
                        point(1.0, slice0, 1.0),
                    ],
                    metadata.clone(),
                ));
            }

            // For the side wall, we normally build a quad spanning from the bottom ring (stack=0)
            // to the top ring (stack=1). If one of the rings is degenerate, that ring reduces to a single point.
            // In that case, we output a triangle.
            if bottom_degenerate {
                // Bottom is a point (start_v); create a triangle from start_v to two consecutive points on the top ring.
                polygons.push(Polygon::new(
                    vec![
                        start_v.clone(),
                        point(1.0, slice0, 0.0),
                        point(1.0, slice1, 0.0),
                    ],
                    metadata.clone(),
                ));
            } else if top_degenerate {
                // Top is a point (end_v); create a triangle from two consecutive points on the bottom ring to end_v.
                polygons.push(Polygon::new(
                    vec![
                        point(0.0, slice1, 0.0),
                        point(0.0, slice0, 0.0),
                        end_v.clone(),
                    ],
                    metadata.clone(),
                ));
            } else {
                // Normal case: both rings are non-degenerate. Use a quad for the side wall.
                polygons.push(Polygon::new(
                    vec![
                        point(0.0, slice1, 0.0),
                        point(0.0, slice0, 0.0),
                        point(1.0, slice0, 0.0),
                        point(1.0, slice1, 0.0),
                    ],
                    metadata.clone(),
                ));
            }
        }

        CSG::from_polygons(&polygons)
    }

    // A helper to create a vertical cylinder along Z from z=0..z=height
    // with the specified radius (NOT diameter).
    pub fn frustrum(radius1: Real, radius2: Real, height: Real, segments: usize, metadata: Option<S>) -> CSG<S> {
        CSG::frustrum_ptp(
            Point3::origin(),
            Point3::new(0.0, 0.0, height),
            radius1,
            radius2,
            segments,
            metadata,
        )
    }

    /// Creates a CSG polyhedron from raw vertex data (`points`) and face indices.
    ///
    /// # Parameters
    ///
    /// - `points`: a slice of `[x,y,z]` coordinates.
    /// - `faces`: each element is a list of indices into `points`, describing one face.
    ///   Each face must have at least 3 indices.
    ///
    /// # Example
    /// ```
    /// # use csgrs::csg::CSG;
    /// let pts = &[
    ///     [0.0, 0.0, 0.0], // point0
    ///     [1.0, 0.0, 0.0], // point1
    ///     [1.0, 1.0, 0.0], // point2
    ///     [0.0, 1.0, 0.0], // point3
    ///     [0.5, 0.5, 1.0], // point4 - top
    /// ];
    ///
    /// // Two faces: bottom square [0,1,2,3], and a pyramid side [0,1,4]
    /// let fcs = vec![
    ///     vec![0, 1, 2, 3],
    ///     vec![0, 1, 4],
    ///     vec![1, 2, 4],
    ///     vec![2, 3, 4],
    ///     vec![3, 0, 4],
    /// ];
    ///
    /// let csg_poly: CSG<f64> = CSG::polyhedron(pts, &fcs, None);
    /// ```
    pub fn polyhedron(points: &[[Real; 3]], faces: &[Vec<usize>], metadata: Option<S>) -> CSG<S> {
        let mut polygons = Vec::new();

        for face in faces {
            // Skip degenerate faces
            if face.len() < 3 {
                continue;
            }

            // Gather the vertices for this face
            let mut face_vertices = Vec::with_capacity(face.len());
            for &idx in face {
                // Ensure the index is valid
                if idx >= points.len() {
                    // todo return error
                    panic!(
                        "Face index {} is out of range (points.len = {}).",
                        idx,
                        points.len()
                    );
                }
                let [x, y, z] = points[idx];
                face_vertices.push(Vertex::new(
                    Point3::new(x, y, z),
                    Vector3::zeros(), // we'll set this later
                ));
            }

            // Build the polygon (plane is auto-computed from first 3 vertices).
            let mut poly = Polygon::new(face_vertices, metadata.clone());

            // Set each vertex normal to match the polygon’s plane normal,
            let plane_normal = poly.plane.normal;
            for v in &mut poly.vertices {
                v.normal = plane_normal;
            }
            polygons.push(poly);
        }

        CSG::from_polygons(&polygons)
    }

    /// # _EGG_
    /// ```js
    /// let egg = undefined
    ///
    /// if (4 == 5)
    ///     alert("no egg")
    /// else {
    ///     console.log(JSON.stringify("egg"))
    /// }
    /// ```
    pub const fn egg(
        _width: Real,
        _length: Real,
        _revolve_segments: u16,
        _outline_segments: i128,
        _metadata: Option<S>,
    ) -> Self {
        panic!("egg");
    }

    /// Apply an arbitrary 3D transform (as a 4x4 matrix) to both polygons and polylines.
    /// The polygon z-coordinates and normal vectors are fully transformed in 3D,
    /// and the 2D polylines are updated by ignoring the resulting z after transform.
    pub fn transform(&self, mat: &Matrix4<Real>) -> CSG<S> {
        let mat_inv_transpose = mat
            .try_inverse().expect("Matrix not invertible?")
            .transpose(); // todo catch error
        let mut csg = self.clone();

        for poly in &mut csg.polygons {
            for vert in &mut poly.vertices {
                // Position
                let hom_pos = mat * vert.pos.to_homogeneous();
                vert.pos = Point3::from_homogeneous(hom_pos).unwrap(); // todo catch error

                // Normal
                vert.normal = mat_inv_transpose.transform_vector(&vert.normal).normalize();
            }

            if poly.vertices.len() >= 3 {
                poly.plane = Plane::from_points(
                    &poly.vertices[0].pos,
                    &poly.vertices[1].pos,
                    &poly.vertices[2].pos,
                );
            }
        }

        // Convert the top-left 2×2 submatrix + translation of a 4×4 into a geo::AffineTransform
        // The 4x4 looks like:
        //  [ m11  m12  m13  m14 ]
        //  [ m21  m22  m23  m24 ]
        //  [ m31  m32  m33  m34 ]
        //  [ m41  m42  m43  m44 ]
        //
        // For 2D, we use the sub-block:
        //   a = m11,  b = m12,
        //   d = m21,  e = m22,
        //   xoff = m14,
        //   yoff = m24,
        // ignoring anything in z.
        //
        // So the final affine transform in 2D has matrix:
        //   [a   b   xoff]
        //   [d   e   yoff]
        //   [0   0    1  ]
        let a = mat[(0, 0)];
        let b = mat[(0, 1)];
        let xoff = mat[(0, 3)];
        let d = mat[(1, 0)];
        let e = mat[(1, 1)];
        let yoff = mat[(1, 3)];

        let affine2 = AffineTransform::new(a, b, xoff, d, e, yoff);

        // 4) Transform csg.geometry (the GeometryCollection) in 2D
        //    Using geo’s map-coords approach or the built-in AffineOps trait.
        //    Below we use the `AffineOps` trait if you have `use geo::AffineOps;`
        csg.geometry = csg.geometry.affine_transform(&affine2);

        csg
    }

    /// Returns a new CSG translated by x, y, and z.
    #[must_use = "Use the new CSG"]
    pub fn translate(&self, x: Real, y: Real, z: Real) -> CSG<S> {
        self.translate_vector(Vector3::new(x, y, z))
    }

    /// Returns a new CSG translated by vector.
    #[must_use = "Use the new CSG"]
    pub fn translate_vector(&self, vector: Vector3<Real>) -> CSG<S> {
        let translation = Translation3::from(vector);

        // Convert to a Matrix4
        let mat4 = translation.to_homogeneous();
        self.transform(&mat4)
    }

    /// Returns a new CSG translated so that its bounding-box center is at the origin (0,0,0).
    #[must_use = "Use the new CSG"]
    pub fn center(&self) -> Self {
        let aabb = self.bounding_box();

        // Compute the AABB center
        let center_x = (aabb.mins.x + aabb.maxs.x) * 0.5;
        let center_y = (aabb.mins.y + aabb.maxs.y) * 0.5;
        let center_z = (aabb.mins.z + aabb.maxs.z) * 0.5;

        // Translate so that the bounding-box center goes to the origin
        self.translate(-center_x, -center_y, -center_z)
    }

    /// Rotates the CSG by `x_degrees`, `y_degrees`, `z_degrees`
    #[must_use = "Use the new CSG"]
    pub fn rotate(&self, x_deg: Real, y_deg: Real, z_deg: Real) -> CSG<S> {
        let rx = Rotation3::from_axis_angle(&Vector3::x_axis(), x_deg.to_radians());
        let ry = Rotation3::from_axis_angle(&Vector3::y_axis(), y_deg.to_radians());
        let rz = Rotation3::from_axis_angle(&Vector3::z_axis(), z_deg.to_radians());

        // Compose them in the desired order
        let rot = rz * ry * rx;
        self.transform(&rot.to_homogeneous())
    }

    /// Scales the CSG by `scale_x`, `scale_y`, `scale_z`
    #[must_use = "Use the new CSG"]
    pub fn scale(&self, sx: Real, sy: Real, sz: Real) -> CSG<S> {
        let mat4 = Matrix4::new_nonuniform_scaling(&Vector3::new(sx, sy, sz));
        self.transform(&mat4)
    }

    /// Reflect (mirror) this CSG about an arbitrary plane `plane`.
    ///
    /// The plane is specified by:
    ///   `plane.normal` = the plane’s normal vector (need not be unit),
    ///   `plane.w`      = the dot-product with that normal for points on the plane (offset).
    ///
    /// Returns a new CSG whose geometry is mirrored accordingly.
    #[must_use = "Use the new CSG"]
    pub fn mirror(&self, plane: &Plane) -> Self {
        // Normal might not be unit, so compute its length:
        let len = plane.normal.norm();
        if len.abs() < EPSILON {
            // Degenerate plane? Just return clone (no transform)
            return self.clone();
        }

        // Unit normal:
        let n = plane.normal / len;
        // Adjusted offset = w / ||n||
        let w = plane.w / len;

        // Step 1) Translate so the plane crosses the origin
        // The plane’s offset vector from origin is (w * n).
        let offset = n * w;
        let t1 = Translation3::from(-offset).to_homogeneous(); // push the plane to origin

        // Step 2) Build the reflection matrix about a plane normal n at the origin
        //   R = I - 2 n n^T
        let mut reflect_4 = Matrix4::identity();
        let reflect_3 = Matrix3::identity() - 2.0 * n * n.transpose();
        reflect_4.fixed_view_mut::<3, 3>(0, 0).copy_from(&reflect_3);

        // Step 3) Translate back
        let t2 = Translation3::from(offset).to_homogeneous(); // pull the plane back out

        // Combine into a single 4×4
        let mirror_mat = t2 * reflect_4 * t1;

        // Apply to all polygons
        self.transform(&mirror_mat).inverse()
    }

    /// Distribute this CSG `count` times along a straight line (vector),
    /// each copy spaced by `spacing`.
    /// E.g. if `dir=(1.0,0.0,0.0)` and `spacing=2.0`, you get copies at
    /// x=0, x=2, x=4, ... etc.
    #[must_use = "Use the new CSG"]
    pub fn distribute_linear(
        &self,
        count: usize,
        dir: nalgebra::Vector3<Real>,
        spacing: Real,
    ) -> CSG<S> {
        if count < 1 {
            return self.clone();
        }
        let step = dir.normalize() * spacing;

        // create a container to hold our unioned copies
        let mut all_csg = CSG::<S>::new();

        for i in 0..count {
            let offset = step * (i as Real);
            let trans = nalgebra::Translation3::from(offset).to_homogeneous();

            // Transform a copy of self and union with other copies
            all_csg = all_csg.union(&self.transform(&trans));
        }

        // Put it in a new CSG
        CSG {
            polygons: all_csg.polygons,
            geometry: all_csg.geometry,
            metadata: self.metadata.clone(),
        }
    }

    /// Distribute this CSG in a grid of `rows x cols`, with spacing dx, dy in XY plane.
    /// top-left or bottom-left depends on your usage of row/col iteration.
    #[must_use = "Use the new CSG"]
    pub fn distribute_grid(&self, rows: usize, cols: usize, dx: Real, dy: Real) -> CSG<S> {
        if rows < 1 || cols < 1 {
            return self.clone();
        }
        let step_x = nalgebra::Vector3::new(dx, 0.0, 0.0);
        let step_y = nalgebra::Vector3::new(0.0, dy, 0.0);

        // create a container to hold our unioned copies
        let mut all_csg = CSG::<S>::new();

        for r in 0..rows {
            for c in 0..cols {
                let offset = step_x * (c as Real) + step_y * (r as Real);
                let trans = nalgebra::Translation3::from(offset).to_homogeneous();

                // Transform a copy of self and union with other copies
                all_csg = all_csg.union(&self.transform(&trans));
            }
        }

        // Put it in a new CSG
        CSG {
            polygons: all_csg.polygons,
            geometry: all_csg.geometry,
            metadata: self.metadata.clone(),
        }
    }

    /// Compute the convex hull of all vertices in this CSG.
    #[cfg(feature = "chull-io")]
    #[must_use = "Use the new CSG"]
    pub fn convex_hull(&self) -> CSG<S> {
        // Gather all (x, y, z) coordinates from the polygons
        let points: Vec<Vec<Real>> = self
            .polygons
            .iter()
            .flat_map(|poly| {
                poly.vertices
                    .iter()
                    .map(|v| vec![v.pos.x, v.pos.y, v.pos.z])
            })
            .collect();

        // Attempt to compute the convex hull using the robust wrapper
        let hull = match ConvexHullWrapper::try_new(&points, None) {
            Ok(h) => h,
            Err(_) => {
                // Fallback to an empty CSG if hull generation fails
                return CSG::new();
            }
        };

        let (verts, indices) = hull.vertices_indices();

        // Reconstruct polygons as triangles
        let mut polygons = Vec::new();
        for tri in indices.chunks(3) {
            let v0 = &verts[tri[0]];
            let v1 = &verts[tri[1]];
            let v2 = &verts[tri[2]];
            let vv0 = Vertex::new(Point3::new(v0[0], v0[1], v0[2]), Vector3::zeros());
            let vv1 = Vertex::new(Point3::new(v1[0], v1[1], v1[2]), Vector3::zeros());
            let vv2 = Vertex::new(Point3::new(v2[0], v2[1], v2[2]), Vector3::zeros());
            polygons.push(Polygon::new(vec![vv0, vv1, vv2], None));
        }

        CSG::from_polygons(&polygons)
    }

    /// Compute the Minkowski sum: self ⊕ other
    ///
    /// Naive approach: Take every vertex in `self`, add it to every vertex in `other`,
    /// then compute the convex hull of all resulting points.
    #[cfg(feature = "chull-io")]
    pub fn minkowski_sum(&self, other: &CSG<S>) -> CSG<S> {
        // Collect all vertices (x, y, z) from self
        let verts_a: Vec<Point3<Real>> = self
            .polygons
            .iter()
            .flat_map(|poly| poly.vertices.iter().map(|v| v.pos))
            .collect();

        // Collect all vertices from other
        let verts_b: Vec<Point3<Real>> = other
            .polygons
            .iter()
            .flat_map(|poly| poly.vertices.iter().map(|v| v.pos))
            .collect();

        if verts_a.is_empty() || verts_b.is_empty() {
            // Empty input to minkowski sum
        }

        // For Minkowski, add every point in A to every point in B
        let sum_points: Vec<_> = verts_a
            .iter()
            .flat_map(|a| verts_b.iter().map(move |b| a + b.coords))
            .map(|v| vec![v.x, v.y, v.z])
            .collect();

        // Compute the hull of these Minkowski-sum points
        let hull = ConvexHullWrapper::try_new(&sum_points, None)
            .expect("Failed to compute Minkowski sum hull");
        let (verts, indices) = hull.vertices_indices();

        // Reconstruct polygons
        let mut polygons = Vec::new();
        for tri in indices.chunks(3) {
            let v0 = &verts[tri[0]];
            let v1 = &verts[tri[1]];
            let v2 = &verts[tri[2]];
            let vv0 = Vertex::new(Point3::new(v0[0], v0[1], v0[2]), Vector3::zeros());
            let vv1 = Vertex::new(Point3::new(v1[0], v1[1], v1[2]), Vector3::zeros());
            let vv2 = Vertex::new(Point3::new(v2[0], v2[1], v2[2]), Vector3::zeros());
            polygons.push(Polygon::new(vec![vv0, vv1, vv2], None));
        }

        CSG::from_polygons(&polygons)
    }

    /// Subdivide all polygons in this CSG 'levels' times, returning a new CSG.
    /// This results in a triangular mesh with more detail.
    #[must_use = "Use the new CSG"]
    pub fn subdivide_triangles(&self, levels: u32) -> CSG<S> {
        if levels == 0 {
            return self.clone();
        }

        #[cfg(feature = "parallel")]
        let new_polygons: Vec<Polygon<S>> = self
            .polygons
            .par_iter()
            .flat_map(|poly| {
                let sub_tris = poly.subdivide_triangles(levels);
                // Convert each small tri back to a Polygon
                sub_tris.into_par_iter().map(move |tri| {
                    Polygon::new(
                        vec![tri[0].clone(), tri[1].clone(), tri[2].clone()],
                        poly.metadata.clone(),
                    )
                })
            })
            .collect();

        #[cfg(not(feature = "parallel"))]
        let new_polygons: Vec<Polygon<S>> = self
            .polygons
            .iter()
            .flat_map(|poly| {
                let sub_tris = poly.subdivide_triangles(levels);
                sub_tris.into_iter().map(move |tri| {
                    Polygon::new(
                        vec![tri[0].clone(), tri[1].clone(), tri[2].clone()],
                        poly.metadata.clone(),
                    )
                })
            })
            .collect();

        CSG::from_polygons(&new_polygons)
    }

    /// Renormalize all polygons in this CSG by re-computing each polygon’s plane
    /// and assigning that plane’s normal to all vertices.
    pub fn renormalize(&mut self) {
        for poly in &mut self.polygons {
            poly.set_new_normal();
        }
    }

    /// Casts a ray defined by `origin` + t * `direction` against all triangles
    /// of this CSG and returns a list of (intersection_point, distance),
    /// sorted by ascending distance.
    ///
    /// # Parameters
    /// - `origin`: The ray’s start point.
    /// - `direction`: The ray’s direction vector.
    ///
    /// # Returns
    /// A `Vec` of `(Point3<Real>, Real)` where:
    /// - `Point3<Real>` is the intersection coordinate in 3D,
    /// - `Real` is the distance (the ray parameter t) from `origin`.
    pub fn ray_intersections(
        &self,
        origin: &Point3<Real>,
        direction: &Vector3<Real>,
    ) -> Vec<(Point3<Real>, Real)> {
        let ray = Ray::new(*origin, *direction);
        let iso = Isometry3::identity(); // No transformation on the triangles themselves.

        let mut hits = Vec::new();

        // 1) For each polygon in the CSG:
        for poly in &self.polygons {
            // 2) Triangulate it if necessary:
            let triangles = poly.tessellate();

            // 3) For each triangle, do a ray–triangle intersection test:
            for tri in triangles {
                let a = tri[0].pos;
                let b = tri[1].pos;
                let c = tri[2].pos;

                // Construct a parry Triangle shape from the 3 vertices:
                let triangle = Triangle::new(a, b, c);

                // Ray-cast against the triangle:
                if let Some(hit) = triangle.cast_ray_and_get_normal(&iso, &ray, Real::MAX, true) {
                    let point_on_ray = ray.point_at(hit.time_of_impact);
                    hits.push((Point3::from(point_on_ray.coords), hit.time_of_impact));
                }
            }
        }

        // 4) Sort hits by ascending distance (toi):
        hits.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        // 5) remove duplicate hits if they fall within tolerance
        hits.dedup_by(|a, b| (a.1 - b.1).abs() < EPSILON);

        hits
    }

    /// Linearly extrude this (2D) shape in the +Z direction by `height`.
    ///
    /// This is just a convenience wrapper around extrude_vector using `Vector3::new(0.0, 0.0, height)`
    pub fn extrude(&self, height: Real) -> CSG<S> {
        self.extrude_vector(Vector3::new(0.0, 0.0, height))
    }

    /// Linearly extrude any 2D geometry (Polygons, MultiPolygons, or sub-geometries
    /// in a GeometryCollection) in `self.geometry` along the given `direction`.
    ///
    /// Builds top, bottom, and side polygons in 3D, storing them in `csg.polygons`.
    /// Returns a new CSG containing these extruded polygons (plus any existing 3D polygons
    /// you already had in `self.polygons`).
    pub fn extrude_vector(&self, direction: Vector3<Real>) -> CSG<S> {
        // If the direction is near zero length, nothing to extrude:
        if direction.norm() < EPSILON {
            return self.clone(); // or return an empty CSG
        }

        // Collect the new 3D polygons generated by extrusion:
        let mut new_polygons = Vec::new();

        // A helper to handle any Geometry
        fn extrude_geometry<S: Clone + Send + Sync>(
            geom: &geo::Geometry<Real>,
            direction: Vector3<Real>,
            metadata: &Option<S>,
            out_polygons: &mut Vec<Polygon<S>>,
        ) {
            match geom {
                geo::Geometry::Polygon(poly) => {
                    let exterior_coords: Vec<[Real; 2]> =
                        poly.exterior().coords_iter().map(|c| [c.x, c.y]).collect();
                    let interior_rings: Vec<Vec<[Real; 2]>> = poly
                        .interiors()
                        .iter()
                        .map(|ring| ring.coords_iter().map(|c| [c.x, c.y]).collect())
                        .collect();

                    // bottom
                    let bottom_tris = CSG::<()>::tessellate_2d(
                        &exterior_coords,
                        &interior_rings.iter().map(|r| &r[..]).collect::<Vec<_>>(),
                    );
                    for tri in bottom_tris {
                        let v0 = Vertex::new(tri[2], -Vector3::z());
                        let v1 = Vertex::new(tri[1], -Vector3::z());
                        let v2 = Vertex::new(tri[0], -Vector3::z());
                        out_polygons.push(Polygon::new(vec![v0, v1, v2], metadata.clone()));
                    }
                    // top
                    let top_tris = CSG::<()>::tessellate_2d(
                        &exterior_coords,
                        &interior_rings.iter().map(|r| &r[..]).collect::<Vec<_>>(),
                    );
                    for tri in top_tris {
                        let p0 = tri[0] + direction;
                        let p1 = tri[1] + direction;
                        let p2 = tri[2] + direction;
                        let v0 = Vertex::new(p0, Vector3::z());
                        let v1 = Vertex::new(p1, Vector3::z());
                        let v2 = Vertex::new(p2, Vector3::z());
                        out_polygons.push(Polygon::new(vec![v0, v1, v2], metadata.clone()));
                    }

                    // sides
                    let all_rings = std::iter::once(poly.exterior()).chain(poly.interiors());
                    for ring in all_rings {
                        let coords: Vec<_> = ring.coords_iter().collect();
                        let n = coords.len();
                        if n < 2 {
                            continue;
                        }
                        for i in 0..(n - 1) {
                            let j = i + 1;
                            let c_i = coords[i];
                            let c_j = coords[j];
                            let b_i = Point3::new(c_i.x, c_i.y, 0.0);
                            let b_j = Point3::new(c_j.x, c_j.y, 0.0);
                            let t_i = b_i + direction;
                            let t_j = b_j + direction;
                            out_polygons.push(Polygon::new(
                                vec![
                                    Vertex::new(b_i, Vector3::zeros()),
                                    Vertex::new(b_j, Vector3::zeros()),
                                    Vertex::new(t_j, Vector3::zeros()),
                                    Vertex::new(t_i, Vector3::zeros()),
                                ],
                                metadata.clone(),
                            ));
                        }
                    }
                }
                geo::Geometry::MultiPolygon(mp) => {
                    for poly in &mp.0 {
                        extrude_geometry(
                            &geo::Geometry::Polygon(poly.clone()),
                            direction,
                            metadata,
                            out_polygons,
                        );
                    }
                }
                geo::Geometry::GeometryCollection(gc) => {
                    for sub in &gc.0 {
                        extrude_geometry(sub, direction, metadata, out_polygons);
                    }
                }
                // Other geometry types (LineString, Point, etc.) are skipped or could be handled differently:
                _ => { /* skip */ }
            }
        }

        // -- 1) Extrude the polygons from self.polygons if you also want to re-extrude them.
        //    (Often, your `polygons` are already 3D, so you might skip extruding them again.)
        //    We'll skip it here for clarity.  If you do want to extrude existing 2D polygons
        //    in `self.polygons`, you'd add similar logic.

        // -- 2) Extrude from self.geometry (the `geo::GeometryCollection`).
        for geom in &self.geometry {
            extrude_geometry(geom, direction, &self.metadata, &mut new_polygons);
        }

        // Combine new extruded polygons with any existing 3D polygons:
        let mut final_polygons = self.polygons.clone();
        final_polygons.extend(new_polygons);

        // Return a new CSG
        CSG {
            polygons: final_polygons,
            geometry: self.geometry.clone(),
            metadata: self.metadata.clone(),
        }
    }

    /// Extrudes (or "lofts") a closed 3D volume between two polygons in space.
    /// - `bottom` and `top` each have the same number of vertices `n`, in matching order.
    /// - Returns a new CSG whose faces are:
    ///   - The `bottom` polygon,
    ///   - The `top` polygon,
    ///   - `n` rectangular side polygons bridging each edge of `bottom` to the corresponding edge of `top`.
    pub fn extrude_between(
        bottom: &Polygon<S>,
        top: &Polygon<S>,
        flip_bottom_polygon: bool,
    ) -> CSG<S> {
        let n = bottom.vertices.len();
        assert_eq!(
            n,
            top.vertices.len(),
            "extrude_between: both polygons must have the same number of vertices" // todo: return error
        );

        // Conditionally flip the bottom polygon if requested.
        let bottom_poly = if flip_bottom_polygon {
            let mut flipped = bottom.clone();
            flipped.flip();
            flipped
        } else {
            bottom.clone()
        };

        // Gather polygons: bottom + top
        // (Depending on the orientation, you might want to flip one of them.)

        let mut polygons = vec![bottom_poly.clone(), top.clone()];

        // For each edge (i -> i+1) in bottom, connect to the corresponding edge in top.
        for i in 0..n {
            let j = (i + 1) % n;
            let b_i = &bottom.vertices[i];
            let b_j = &bottom.vertices[j];
            let t_i = &top.vertices[i];
            let t_j = &top.vertices[j];

            // Build the side face as a 4-vertex polygon (quad).
            // Winding order here is chosen so that the polygon's normal faces outward
            // (depending on the orientation of bottom vs. top).
            let side_poly = Polygon::new(
                vec![
                    b_i.clone(), // bottom[i]
                    b_j.clone(), // bottom[i+1]
                    t_j.clone(), // top[i+1]
                    t_i.clone(), // top[i]
                ],
                bottom.metadata.clone(), // carry over bottom polygon metadata
            );
            polygons.push(side_poly);
        }

        CSG::from_polygons(&polygons)
    }

    /// Returns a [`parry3d::bounding_volume::Aabb`] by merging:
    /// 1. The 3D bounds of all `polygons`.
    /// 2. The 2D bounding rectangle of `self.geometry`, interpreted at z=0.
    pub fn bounding_box(&self) -> Aabb {
        // Track overall min/max in x, y, z among all 3D polygons and the 2D geometry’s bounding_rect.
        let mut min_x = Real::MAX;
        let mut min_y = Real::MAX;
        let mut min_z = Real::MAX;
        let mut max_x = -Real::MAX;
        let mut max_y = -Real::MAX;
        let mut max_z = -Real::MAX;

        // 1) Gather from the 3D polygons
        for poly in &self.polygons {
            for v in &poly.vertices {
                min_x = *partial_min(&min_x, &v.pos.x).unwrap();
                min_y = *partial_min(&min_y, &v.pos.y).unwrap();
                min_z = *partial_min(&min_z, &v.pos.z).unwrap();

                max_x = *partial_max(&max_x, &v.pos.x).unwrap();
                max_y = *partial_max(&max_y, &v.pos.y).unwrap();
                max_z = *partial_max(&max_z, &v.pos.z).unwrap();
            }
        }

        // 2) Gather from the 2D geometry using `geo::BoundingRect`
        //    This gives us (min_x, min_y) / (max_x, max_y) in 2D. For 3D, treat z=0.
        //    Explicitly capture the result of `.bounding_rect()` as an Option<Rect<Real>>
        let maybe_rect: Option<Rect<Real>> = self.geometry.bounding_rect();
        if let Some(rect) = maybe_rect {
            let min_pt = rect.min();
            let max_pt = rect.max();

            // Merge the 2D bounds into our existing min/max, forcing z=0 for 2D geometry.
            min_x = *partial_min(&min_x, &min_pt.x).unwrap();
            min_y = *partial_min(&min_y, &min_pt.y).unwrap();
            min_z = *partial_min(&min_z, &0.0).unwrap();

            max_x = *partial_max(&max_x, &max_pt.x).unwrap();
            max_y = *partial_max(&max_y, &max_pt.y).unwrap();
            max_z = *partial_max(&max_z, &0.0).unwrap();
        }

        // If still uninitialized (e.g., no polygons or geometry), return a trivial AABB at origin
        if min_x > max_x {
            return Aabb::new(Point3::origin(), Point3::origin());
        }

        // Build a parry3d Aabb from these min/max corners
        let mins = Point3::new(min_x, min_y, min_z);
        let maxs = Point3::new(max_x, max_y, max_z);
        Aabb::new(mins, maxs)
    }

    /// Grows/shrinks/offsets all polygons in the XY plane by `distance` using cavalier_contours `parallel_offset`.
    /// for each Polygon we convert to a `cavalier_contours` `Polyline<Real>` and call `parallel_offset`
    #[cfg(feature = "offset")]
    pub fn offset(&self, distance: Real) -> CSG<S> {
        // For each Geometry in the collection:
        //   - If it's a Polygon, buffer it and store the result as a MultiPolygon
        //   - If it's a MultiPolygon, buffer it directly
        //   - Otherwise, ignore (exclude) it from the new collection
        let offset_geoms = self
            .geometry
            .iter()
            .filter_map(|geom| match geom {
                Geometry::Polygon(poly) => {
                    let new_mpoly = buffer_polygon(poly, distance);
                    Some(Geometry::MultiPolygon(new_mpoly))
                }
                Geometry::MultiPolygon(mpoly) => {
                    let new_mpoly = buffer_multi_polygon(mpoly, distance);
                    Some(Geometry::MultiPolygon(new_mpoly))
                }
                _ => None, // ignore other geometry types
            })
            .collect();

        // Construct a new GeometryCollection from the offset geometries
        let new_collection = GeometryCollection(offset_geoms);

        // Return a new CSG using the offset geometry collection and the old polygons/metadata
        CSG {
            polygons: self.polygons.clone(),
            geometry: new_collection,
            metadata: self.metadata.clone(),
        }
    }

    /// Flattens any 3D polygons by projecting them onto the XY plane (z=0),
    /// unifies them into one or more 2D polygons, and returns a purely 2D CSG.
    ///
    /// - If this CSG is already 2D (`self.polygons` is empty), just returns `self.clone()`.
    /// - Otherwise, all `polygons` are tessellated, projected into XY, and unioned.
    /// - We also union any existing 2D geometry (`self.geometry`).
    /// - The output has `.polygons` empty and `.geometry` containing the final 2D shape.
    #[must_use = "Use the new CSG"]
    pub fn flatten(&self) -> CSG<S> {
        // 1) If there are no 3D polygons, this is already purely 2D => return as-is
        if self.polygons.is_empty() {
            return self.clone();
        }

        // 2) Convert all 3D polygons into a collection of 2D polygons
        let mut flattened_3d = Vec::new(); // will store geo::Polygon<Real>

        for poly in &self.polygons {
            // Tessellate this polygon into triangles
            let triangles = poly.tessellate();
            // Each triangle has 3 vertices [v0, v1, v2].
            // Project them onto XY => build a 2D polygon (triangle).
            for tri in triangles {
                let ring = vec![
                    (tri[0].pos.x, tri[0].pos.y),
                    (tri[1].pos.x, tri[1].pos.y),
                    (tri[2].pos.x, tri[2].pos.y),
                    (tri[0].pos.x, tri[0].pos.y), // close ring explicitly
                ];
                let polygon_2d = geo::Polygon::new(LineString::from(ring), vec![]);
                flattened_3d.push(polygon_2d);
            }
        }

        // 3) Union all these polygons together into one MultiPolygon
        //    (We could chain them in a fold-based union.)
        let unioned_from_3d = if flattened_3d.is_empty() {
            MultiPolygon::new(Vec::new())
        } else {
            // Start with the first polygon as a MultiPolygon
            let mut mp_acc = MultiPolygon(vec![flattened_3d[0].clone()]);
            // Union in the rest
            for p in flattened_3d.iter().skip(1) {
                mp_acc = mp_acc.union(&MultiPolygon(vec![p.clone()]));
            }
            mp_acc
        };

        // 4) Union this with any existing 2D geometry (polygons) from self.geometry
        let existing_2d = gc_to_polygons(&self.geometry); // turns geometry -> MultiPolygon
        let final_union = unioned_from_3d.union(&existing_2d);
        // Optionally ensure consistent orientation (CCW for exteriors):
        let oriented = final_union.orient(Direction::Default);

        // 5) Store final polygons as a MultiPolygon in a new GeometryCollection
        let mut new_gc = GeometryCollection::default();
        new_gc.0.push(Geometry::MultiPolygon(oriented));

        // 6) Return a purely 2D CSG: polygons empty, geometry has the final shape
        CSG {
            polygons: Vec::new(),
            geometry: new_gc,
            metadata: self.metadata.clone(),
        }
    }

    /// Slice this solid by a given `plane`, returning a new `CSG` whose polygons
    /// are either:
    /// - The polygons that lie exactly in the slicing plane (coplanar), or
    /// - Polygons formed by the intersection edges (each a line, possibly open or closed).
    ///
    /// The returned `CSG` can contain:
    /// - **Closed polygons** that are coplanar,
    /// - **Open polygons** (poly-lines) if the plane cuts through edges,
    /// - Potentially **closed loops** if the intersection lines form a cycle.
    ///
    /// # Example
    /// ```
    /// let cylinder = CSG::cylinder(1.0, 2.0, 32, None);
    /// let plane_z0 = Plane { normal: Vector3::z(), w: 0.0 };
    /// let cross_section = cylinder.slice(plane_z0);
    /// // `cross_section` will contain:
    /// //   - Possibly an open or closed polygon(s) at z=0
    /// //   - Or empty if no intersection
    /// ```
    #[cfg(feature = "hashmap")]
    #[must_use = "Use the new CSG"]
    pub fn slice(&self, plane: &Plane) -> CSG<S> {
        // Build a BSP from all of our polygons:
        let node = Node::new(&self.polygons.clone());

        // Ask the BSP for coplanar polygons + intersection edges:
        let (coplanar_polys, intersection_edges) = node.slice(plane);

        // “Knit” those intersection edges into polylines. Each edge is [vA, vB].
        let polylines_3d = unify_intersection_edges(&intersection_edges);

        // Convert each polyline of vertices into a Polygon<S>
        let mut result_polygons = Vec::new();

        // Add the coplanar polygons. We can re‐assign their plane to `plane` to ensure
        // they share the exact plane definition (in case of numeric drift).
        for mut p in coplanar_polys {
            p.plane = plane.clone(); // unify plane data
            result_polygons.push(p);
        }

        let mut new_gc = GeometryCollection::default();

        // Convert the “chains” or loops into open/closed polygons
        for mut chain in polylines_3d {
            let n = chain.len();
            if n < 2 {
                // degenerate
                continue;
            }

            // check if first and last point are within EPSILON of each other
            let dist_sq = (chain[0].pos - chain[n - 1].pos).norm_squared();
            if dist_sq < EPSILON * EPSILON {
                // Force them to be exactly the same, closing the line
                chain[n - 1] = chain[0].clone();
            }

            let polyline = LineString::new(
                    chain.iter()
                        .map(|vertex| {coord! {x: vertex.pos.x, y: vertex.pos.y}})
                        .collect()
                );

            if polyline.is_closed() {
                let polygon = GeoPolygon::new(polyline, vec![]);
                let oriented = polygon.orient(Direction::Default);
                new_gc.0.push(Geometry::Polygon(oriented));
            } else {
                new_gc.0.push(Geometry::LineString(polyline));
            }
        }

        // Return a purely 2D CSG: polygons empty, geometry has the final shape
        CSG {
            polygons: Vec::new(),
            geometry: new_gc,
            metadata: self.metadata.clone(),
        }
    }

    /// Triangulate each polygon in the CSG returning a CSG containing triangles
    #[must_use = "Use the new CSG"]
    pub fn tessellate(&self) -> CSG<S> {
        let mut triangles = Vec::new();

        for poly in &self.polygons {
            let tris = poly.tessellate();
            for triangle in tris {
                triangles.push(Polygon::new(triangle.to_vec(), poly.metadata.clone()));
            }
        }

        CSG::from_polygons(&triangles)
    }

    /// Convert the polygons in this `CSG` to a Parry `TriMesh`.\
    /// Useful for collision detection or physics simulations.
    pub fn to_trimesh(&self) -> SharedShape {
        // 1) Gather all the triangles from each polygon
        // 2) Build a TriMesh from points + triangle indices
        // 3) Wrap that in a SharedShape to be used in Rapier
        let tri_csg = self.tessellate();
        let mut vertices = Vec::new();
        let mut indices = Vec::new();
        let mut index_offset = 0;

        for poly in &tri_csg.polygons {
            let a = poly.vertices[0].pos;
            let b = poly.vertices[1].pos;
            let c = poly.vertices[2].pos;

            vertices.push(a);
            vertices.push(b);
            vertices.push(c);

            indices.push([index_offset, index_offset + 1, index_offset + 2]);
            index_offset += 3;
        }

        // TriMesh::new(Vec<[Real; 3]>, Vec<[u32; 3]>)
        let trimesh = TriMesh::new(vertices, indices).unwrap(); // todo: handle error
        SharedShape::new(trimesh)
    }

    /// Approximate mass properties using Rapier.
    pub fn mass_properties(&self, density: Real) -> (Real, Point3<Real>, Unit<Quaternion<Real>>) {
        let shape = self.to_trimesh();
        if let Some(trimesh) = shape.as_trimesh() {
            let mp = trimesh.mass_properties(density);
            (
                mp.mass(),
                mp.local_com,                     // a Point3<Real>
                mp.principal_inertia_local_frame, // a Unit<Quaternion<Real>>
            )
        } else {
            // fallback if not a TriMesh
            (0.0, Point3::origin(), Unit::<Quaternion<Real>>::identity())
        }
    }

    /// Create a Rapier rigid body + collider from this CSG, using
    /// an axis-angle `rotation` in 3D (the vector’s length is the
    /// rotation in radians, and its direction is the axis).
    pub fn to_rigid_body(
        &self,
        rb_set: &mut RigidBodySet,
        co_set: &mut ColliderSet,
        translation: Vector3<Real>,
        rotation: Vector3<Real>, // rotation axis scaled by angle (radians)
        density: Real,
    ) -> RigidBodyHandle {
        let shape = self.to_trimesh();

        // Build a Rapier RigidBody
        let rb = RigidBodyBuilder::dynamic()
            .translation(translation)
            // Now `rotation(...)` expects an axis-angle Vector3.
            .rotation(rotation)
            .build();
        let rb_handle = rb_set.insert(rb);

        // Build the collider
        let coll = ColliderBuilder::new(shape).density(density).build();
        co_set.insert_with_parent(coll, rb_handle, rb_set);

        rb_handle
    }

    /// Checks if the CSG object is manifold.
    ///
    /// This function defines a comparison function which takes EPSILON into account
    /// for Real coordinates, builds a hashmap key from the string representation of
    /// the coordinates, tessellates the CSG polygons, gathers each of their three edges,
    /// counts how many times each edge appears across all triangles,
    /// and returns true if every edge appears exactly 2 times, else false.
    ///
    /// We should also check that all faces have consistent orientation and no neighbors
    /// have flipped normals.
    ///
    /// We should also check for zero-area triangles
    ///
    /// # Returns
    ///
    /// - `true`: If the CSG object is manifold.
    /// - `false`: If the CSG object is not manifold.
    #[cfg(feature = "hashmap")]
    pub fn is_manifold(&self) -> bool {
        fn approx_lt(a: &Point3<Real>, b: &Point3<Real>) -> bool {
            // Compare x
            if (a.x - b.x).abs() > EPSILON {
                return a.x < b.x;
            }
            // If x is "close", compare y
            if (a.y - b.y).abs() > EPSILON {
                return a.y < b.y;
            }
            // If y is also close, compare z
            a.z < b.z
        }

        // Turn a 3D point into a string with limited decimal places
        fn point_key(p: &Point3<Real>) -> String {
            // Truncate/round to e.g. 6 decimals
            format!("{:.6},{:.6},{:.6}", p.x, p.y, p.z)
        }

        // Triangulate the whole shape once
        let tri_csg = self.tessellate();
        let mut edge_counts: HashMap<(String, String), u32> = HashMap::new();

        for poly in &tri_csg.polygons {
            // Each tri is 3 vertices: [v0, v1, v2]
            // We'll look at edges (0->1, 1->2, 2->0).
            for &(i0, i1) in &[(0, 1), (1, 2), (2, 0)] {
                let p0 = &poly.vertices[i0].pos;
                let p1 = &poly.vertices[i1].pos;

                // Order them so (p0, p1) and (p1, p0) become the same key
                let (a_key, b_key) = if approx_lt(p0, p1) {
                    (point_key(p0), point_key(p1))
                } else {
                    (point_key(p1), point_key(p0))
                };

                *edge_counts.entry((a_key, b_key)).or_insert(0) += 1;
            }
        }

        // For a perfectly closed manifold surface (with no boundary),
        // each edge should appear exactly 2 times.
        edge_counts.values().all(|&count| count == 2)
    }

    /// **Creates a CSG from a list of metaballs** by sampling a 3D grid and using marching cubes.
    ///
    /// - `balls`: slice of metaball definitions (center + radius).
    /// - `resolution`: (nx, ny, nz) defines how many steps along x, y, z.
    /// - `iso_value`: threshold at which the isosurface is extracted.
    /// - `padding`: extra margin around the bounding region (e.g. 0.5) so the surface doesn’t get truncated.
    #[cfg(feature = "metaballs")]
    pub fn metaballs(
        balls: &[MetaBall],
        resolution: (usize, usize, usize),
        iso_value: Real,
        padding: Real,
        metadata: Option<S>,
    ) -> CSG<S> {
        if balls.is_empty() {
            return CSG::new();
        }

        // Determine bounding box of all metaballs (plus padding).
        let mut min_pt = Point3::new(Real::MAX, Real::MAX, Real::MAX);
        let mut max_pt = Point3::new(-Real::MAX, -Real::MAX, -Real::MAX);

        for mb in balls {
            let c = &mb.center;
            let r = mb.radius + padding;

            if c.x - r < min_pt.x {
                min_pt.x = c.x - r;
            }
            if c.y - r < min_pt.y {
                min_pt.y = c.y - r;
            }
            if c.z - r < min_pt.z {
                min_pt.z = c.z - r;
            }

            if c.x + r > max_pt.x {
                max_pt.x = c.x + r;
            }
            if c.y + r > max_pt.y {
                max_pt.y = c.y + r;
            }
            if c.z + r > max_pt.z {
                max_pt.z = c.z + r;
            }
        }

        // Resolution for X, Y, Z
        let nx = resolution.0.max(2) as u32;
        let ny = resolution.1.max(2) as u32;
        let nz = resolution.2.max(2) as u32;

        // Spacing in each axis
        let dx = (max_pt.x - min_pt.x) / (nx as Real - 1.0);
        let dy = (max_pt.y - min_pt.y) / (ny as Real - 1.0);
        let dz = (max_pt.z - min_pt.z) / (nz as Real - 1.0);

        // Create and fill the scalar-field array with "field_value - iso_value"
        // so that the isosurface will be at 0.
        let array_size = (nx * ny * nz) as usize;
        let mut field_values = vec![0f32; array_size];

        let index_3d = |ix: u32, iy: u32, iz: u32| -> usize {
            (iz * ny + iy) as usize * (nx as usize) + ix as usize
        };

        for iz in 0..nz {
            let zf = min_pt.z + (iz as Real) * dz;
            for iy in 0..ny {
                let yf = min_pt.y + (iy as Real) * dy;
                for ix in 0..nx {
                    let xf = min_pt.x + (ix as Real) * dx;
                    let p = Point3::new(xf, yf, zf);

                    let val = scalar_field_metaballs(balls, &p) - iso_value;
                    field_values[index_3d(ix, iy, iz)] = val as f32;
                }
            }
        }

        // Use fast-surface-nets to extract a mesh from this 3D scalar field.
        // We'll define a shape type for ndshape:
        #[allow(non_snake_case)]
        #[derive(Clone, Copy)]
        struct GridShape {
            nx: u32,
            ny: u32,
            nz: u32,
        }

        impl fast_surface_nets::ndshape::Shape<3> for GridShape {
            type Coord = u32;
            #[inline]
            fn as_array(&self) -> [Self::Coord; 3] {
                [self.nx, self.ny, self.nz]
            }

            fn size(&self) -> Self::Coord {
                self.nx * self.ny * self.nz
            }

            fn usize(&self) -> usize {
                (self.nx * self.ny * self.nz) as usize
            }

            fn linearize(&self, coords: [Self::Coord; 3]) -> u32 {
                let [x, y, z] = coords;
                (z * self.ny + y) * self.nx + x
            }

            fn delinearize(&self, i: u32) -> [Self::Coord; 3] {
                let x = i % (self.nx);
                let yz = i / (self.nx);
                let y = yz % (self.ny);
                let z = yz / (self.ny);
                [x, y, z]
            }
        }

        let shape = GridShape { nx, ny, nz };

        // We'll collect the output into a SurfaceNetsBuffer
        let mut sn_buffer = SurfaceNetsBuffer::default();

        // The region we pass to surface_nets is the entire 3D range [0..nx, 0..ny, 0..nz]
        // minus 1 in each dimension to avoid indexing past the boundary:
        let (max_x, max_y, max_z) = (nx - 1, ny - 1, nz - 1);

        surface_nets(
            &field_values, // SDF array
            &shape,        // custom shape
            [0, 0, 0],     // minimum corner in lattice coords
            [max_x, max_y, max_z],
            &mut sn_buffer,
        );

        // Convert the resulting surface net indices/positions into Polygons
        // for the csgrs data structures.
        let mut triangles = Vec::with_capacity(sn_buffer.indices.len() / 3);

        for tri in sn_buffer.indices.chunks_exact(3) {
            let i0 = tri[0] as usize;
            let i1 = tri[1] as usize;
            let i2 = tri[2] as usize;

            let p0_index = sn_buffer.positions[i0];
            let p1_index = sn_buffer.positions[i1];
            let p2_index = sn_buffer.positions[i2];

            // Convert from index space to real (world) space:
            let p0_real = Point3::new(
                min_pt.x + p0_index[0] as Real * dx,
                min_pt.y + p0_index[1] as Real * dy,
                min_pt.z + p0_index[2] as Real * dz,
            );

            let p1_real = Point3::new(
                min_pt.x + p1_index[0] as Real * dx,
                min_pt.y + p1_index[1] as Real * dy,
                min_pt.z + p1_index[2] as Real * dz,
            );

            let p2_real = Point3::new(
                min_pt.x + p2_index[0] as Real * dx,
                min_pt.y + p2_index[1] as Real * dy,
                min_pt.z + p2_index[2] as Real * dz,
            );

            // Likewise for the normals if you want them in true world space.
            // Usually you'd need to do an inverse-transpose transform if your
            // scale is non-uniform. For uniform voxels, scaling is simpler:

            let n0 = sn_buffer.normals[i0];
            let n1 = sn_buffer.normals[i1];
            let n2 = sn_buffer.normals[i2];

            // Construct your vertices:
            let v0 = Vertex::new(p0_real, Vector3::new(n0[0] as Real, n0[1] as Real, n0[2] as Real));
            let v1 = Vertex::new(p1_real, Vector3::new(n1[0] as Real, n1[1] as Real, n1[2] as Real));
            let v2 = Vertex::new(p2_real, Vector3::new(n2[0] as Real, n2[1] as Real, n2[2] as Real));

            // Each tri is turned into a Polygon with 3 vertices
            let poly = Polygon::new(vec![v0, v2, v1], metadata.clone());
            triangles.push(poly);
        }

        // Build and return a CSG from these polygons
        CSG::from_polygons(&triangles)
    }

    /// Return a CSG created by meshing a signed distance field within a bounding box
    ///
    /// ```no_run
    /// # use csgrs::csg::CSG;
    /// // Example SDF for a sphere of radius 1.5 centered at (0,0,0)
    /// let my_sdf = |p: &Point3<Real>| p.coords.norm() - 1.5;
    ///
    /// let resolution = (60, 60, 60);
    /// let min_pt = Point3::new(-2.0, -2.0, -2.0);
    /// let max_pt = Point3::new( 2.0,  2.0,  2.0);
    /// let iso_value = 0.0; // Typically zero for SDF-based surfaces
    ///
    /// let csg_shape = CSG::from_sdf(my_sdf, resolution, min_pt, max_pt, iso_value);
    /// ```
    #[cfg(feature = "sdf")]
    pub fn sdf<F>(
        sdf: F,
        resolution: (usize, usize, usize),
        min_pt: Point3<Real>,
        max_pt: Point3<Real>,
        iso_value: Real,
        metadata: &Option<S>,
    ) -> CSG<S>
    where
        // F is a closure or function that takes a 3D point and returns the signed distance.
        // Must be `Sync`/`Send` if you want to parallelize the sampling.
        F: Fn(&Point3<Real>) -> Real + Sync + Send,
    {
        use crate::float_types::Real;
        use fast_surface_nets::{SurfaceNetsBuffer, surface_nets};

        /// The shape describing our discrete grid for Surface Nets:
        #[derive(Clone, Copy)]
        struct GridShape {
            nx: u32,
            ny: u32,
            nz: u32,
        }

        impl fast_surface_nets::ndshape::Shape<3> for GridShape {
            type Coord = u32;

            #[inline]
            fn as_array(&self) -> [Self::Coord; 3] {
                [self.nx, self.ny, self.nz]
            }

            fn size(&self) -> Self::Coord {
                self.nx * self.ny * self.nz
            }

            fn usize(&self) -> usize {
                (self.nx * self.ny * self.nz) as usize
            }

            fn linearize(&self, coords: [Self::Coord; 3]) -> u32 {
                let [x, y, z] = coords;
                (z * self.ny + y) * self.nx + x
            }

            fn delinearize(&self, i: u32) -> [Self::Coord; 3] {
                let x = i % self.nx;
                let yz = i / self.nx;
                let y = yz % self.ny;
                let z = yz / self.ny;
                [x, y, z]
            }
        }

        // Early return if resolution is degenerate
        let nx = resolution.0.max(2) as u32;
        let ny = resolution.1.max(2) as u32;
        let nz = resolution.2.max(2) as u32;

        // Determine grid spacing based on bounding box and resolution
        let dx = (max_pt.x - min_pt.x) / (nx as Real - 1.0);
        let dy = (max_pt.y - min_pt.y) / (ny as Real - 1.0);
        let dz = (max_pt.z - min_pt.z) / (nz as Real - 1.0);

        // Allocate storage for field values:
        let array_size = (nx * ny * nz) as usize;
        let mut field_values = vec![0.0_f32; array_size];

        // Helper to map (ix, iy, iz) to 1D index:
        let index_3d = |ix: u32, iy: u32, iz: u32| -> usize {
            (iz * ny + iy) as usize * (nx as usize) + ix as usize
        };

        // Sample the SDF at each grid cell:
        // Note that for an "isosurface" at iso_value, we store (sdf_value - iso_value)
        // so that `surface_nets` zero-crossing aligns with iso_value.
        for iz in 0..nz {
            let zf = min_pt.z + (iz as Real) * dz;
            for iy in 0..ny {
                let yf = min_pt.y + (iy as Real) * dy;
                for ix in 0..nx {
                    let xf = min_pt.x + (ix as Real) * dx;
                    let p = Point3::new(xf, yf, zf);
                    let sdf_val = sdf(&p);
                    // Shift by iso_value so that the zero-level is the surface we want:
                    field_values[index_3d(ix, iy, iz)] = (sdf_val - iso_value) as f32;
                }
            }
        }

        let shape = GridShape { nx, ny, nz };

        // `SurfaceNetsBuffer` collects the positions, normals, and triangle indices
        let mut sn_buffer = SurfaceNetsBuffer::default();

        // The max valid coordinate in each dimension
        let max_x = nx - 1;
        let max_y = ny - 1;
        let max_z = nz - 1;

        // Run surface nets
        surface_nets(
            &field_values,
            &shape,
            [0, 0, 0],
            [max_x, max_y, max_z],
            &mut sn_buffer,
        );

        // Convert the resulting triangles into CSG polygons
        let mut triangles = Vec::with_capacity(sn_buffer.indices.len() / 3);

        for tri in sn_buffer.indices.chunks_exact(3) {
            let i0 = tri[0] as usize;
            let i1 = tri[1] as usize;
            let i2 = tri[2] as usize;

            let p0i = sn_buffer.positions[i0];
            let p1i = sn_buffer.positions[i1];
            let p2i = sn_buffer.positions[i2];

            // Convert from [u32; 3] to real coordinates:
            let p0 = Point3::new(
                min_pt.x + p0i[0] as Real * dx,
                min_pt.y + p0i[1] as Real * dy,
                min_pt.z + p0i[2] as Real * dz,
            );
            let p1 = Point3::new(
                min_pt.x + p1i[0] as Real * dx,
                min_pt.y + p1i[1] as Real * dy,
                min_pt.z + p1i[2] as Real * dz,
            );
            let p2 = Point3::new(
                min_pt.x + p2i[0] as Real * dx,
                min_pt.y + p2i[1] as Real * dy,
                min_pt.z + p2i[2] as Real * dz,
            );

            // Retrieve precomputed normal from Surface Nets:
            let n0 = sn_buffer.normals[i0];
            let n1 = sn_buffer.normals[i1];
            let n2 = sn_buffer.normals[i2];

            let v0 = Vertex::new(
                p0,
                Vector3::new(n0[0] as Real, n0[1] as Real, n0[2] as Real),
            );
            let v1 = Vertex::new(
                p1,
                Vector3::new(n1[0] as Real, n1[1] as Real, n1[2] as Real),
            );
            let v2 = Vertex::new(
                p2,
                Vector3::new(n2[0] as Real, n2[1] as Real, n2[2] as Real),
            );

            // Note: reverse v1, v2 if you need to fix winding
            let poly = Polygon::new(vec![v0, v1, v2], metadata.clone());
            triangles.push(poly);
        }

        // Return as a CSG
        CSG::from_polygons(&triangles)
    }
}

/// Helper to build a single Polygon from a “slice” of 3D points.
///
/// If `flip_winding` is true, we reverse the vertex order (so the polygon’s normal flips).
fn _polygon_from_slice<S: Clone + Send + Sync>(
    slice_pts: &[Point3<Real>],
    flip_winding: bool,
    metadata: Option<S>,
) -> Polygon<S> {
    if slice_pts.len() < 3 {
        // degenerate polygon
        return Polygon::new(vec![], metadata);
    }
    // Build the vertex list
    let mut verts: Vec<Vertex> = slice_pts
        .iter()
        .map(|p| Vertex::new(*p, Vector3::zeros()))
        .collect();

    if flip_winding {
        verts.reverse();
        for v in &mut verts {
            v.flip();
        }
    }

    let mut poly = Polygon::new(verts, metadata);
    poly.set_new_normal(); // Recompute its plane & normal for consistency
    poly
}

// Extract only the polygons from a geometry collection
fn gc_to_polygons(gc: &GeometryCollection<Real>) -> MultiPolygon<Real> {
    let mut polygons = vec![];
    for geom in &gc.0 {
        match geom {
            Geometry::Polygon(poly) => polygons.push(poly.clone()),
            Geometry::MultiPolygon(mp) => polygons.extend(mp.0.clone()),
            // ignore lines, points, etc.
            _ => {}
        }
    }
    MultiPolygon(polygons)
}

// Build a small helper for hashing endpoints:
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct EndKey(i64, i64, i64);

/// Round a floating to a grid for hashing
fn quantize(x: Real) -> i64 {
    // For example, scale by 1e8
    (x * 1e8).round() as i64
}

/// Convert a Vertex’s position to an `EndKey`
fn make_key(pos: &Point3<Real>) -> EndKey {
    EndKey(quantize(pos.x), quantize(pos.y), quantize(pos.z))
}

/// Take a list of intersection edges `[Vertex;2]` and merge them into polylines.
/// Each edge is a line segment between two 3D points.  We want to “knit” them together by
/// matching endpoints that lie within EPSILON of each other, forming either open or closed chains.
///
/// This returns a `Vec` of polylines, where each polyline is a `Vec<Vertex>`.
#[cfg(feature = "hashmap")]
fn unify_intersection_edges(edges: &[[Vertex; 2]]) -> Vec<Vec<Vertex>> {
    // We will store adjacency by a “key” that identifies an endpoint up to EPSILON,
    // then link edges that share the same key.

    // Adjacency map: key -> list of (edge_index, is_start_or_end)
    // We’ll store “(edge_idx, which_end)” as which_end = 0 or 1 for edges[edge_idx][0/1].
    let mut adjacency: HashMap<EndKey, Vec<(usize, usize)>> = HashMap::new();

    // Collect all endpoints
    for (i, edge) in edges.iter().enumerate() {
        for end_idx in 0..2 {
            let v = &edge[end_idx];
            let k = make_key(&v.pos);
            adjacency.entry(k).or_default().push((i, end_idx));
        }
    }

    // We’ll keep track of which edges have been “visited” in the final polylines.
    let mut visited = vec![false; edges.len()];

    let mut chains: Vec<Vec<Vertex>> = Vec::new();

    // For each edge not yet visited, we “walk” outward from one end, building a chain
    for start_edge_idx in 0..edges.len() {
        if visited[start_edge_idx] {
            continue;
        }
        // Mark it visited
        visited[start_edge_idx] = true;

        // Our chain starts with `edges[start_edge_idx]`. We can build a small function to “walk”:
        // We’ll store it in the direction edge[0] -> edge[1]
        let e = &edges[start_edge_idx];
        let mut chain = vec![e[0].clone(), e[1].clone()];

        // We walk “forward” from edge[1] if possible
        extend_chain_forward(&mut chain, &adjacency, &mut visited, edges);

        // We also might walk “backward” from edge[0], but
        // we can do that by reversing the chain at the end if needed. Alternatively,
        // we can do a separate pass.  Let’s do it in place for clarity:
        chain.reverse();
        extend_chain_forward(&mut chain, &adjacency, &mut visited, edges);
        // Then reverse back so it goes in the original direction
        chain.reverse();

        chains.push(chain);
    }

    chains
}

/// Extends a chain “forward” by repeatedly finding any unvisited edge that starts
/// at the chain’s current end vertex.
#[cfg(feature = "hashmap")]
fn extend_chain_forward(
    chain: &mut Vec<Vertex>,
    adjacency: &HashMap<EndKey, Vec<(usize, usize)>>,
    visited: &mut [bool],
    edges: &[[Vertex; 2]],
) {
    loop {
        // The chain’s current end point:
        let last_v = chain.last().unwrap();
        let key = make_key(&last_v.pos);

        // Find candidate edges that share this endpoint
        let Some(candidates) = adjacency.get(&key) else {
            break;
        };

        // Among these candidates, we want one whose “other endpoint” we can follow
        // and is not visited yet.
        let mut found_next = None;
        for &(edge_idx, end_idx) in candidates {
            if visited[edge_idx] {
                continue;
            }
            // If this is edges[edge_idx][end_idx], the “other” end is edges[edge_idx][1-end_idx].
            // We want that other end to continue the chain.
            let other_end_idx = 1 - end_idx;
            let next_vertex = &edges[edge_idx][other_end_idx];

            // But we must also confirm that the last_v is indeed edges[edge_idx][end_idx]
            // (within EPSILON) which we have checked via the key, so likely yes.

            // Mark visited
            visited[edge_idx] = true;
            found_next = Some(next_vertex.clone());
            break;
        }

        match found_next {
            Some(v) => {
                chain.push(v);
            }
            None => {
                break;
            }
        }
    }
}

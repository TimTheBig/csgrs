#![allow(dead_code)]
#![forbid(unsafe_code)]

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use nalgebra::{Matrix4, Vector3, Point3, Translation3, Rotation3, Isometry3, Unit, Quaternion};
use chull::ConvexHullWrapper;
use parry3d_f64::{
    bounding_volume::Aabb,
    query::{Ray, RayCast},
    shape::{Triangle, TriMesh, SharedShape},
    };
use rapier3d_f64::prelude::*;
use meshtext::{Glyph, MeshGenerator, MeshText};
use stl_io;
use std::io::Cursor;
use std::fs::OpenOptions;
use cavalier_contours::polyline::{
    Polyline, PlineSource, PlineCreation, PlineSourceMut, BooleanOp, PlineBooleanOptions
};
use cavalier_contours::polyline::internal::pline_boolean::polyline_boolean;
use earclip::earcut;

#[cfg(test)]
mod tests;

const EPSILON: f64 = 1e-5;

pub enum Axis {
    X,
    Y,
    Z,
}

/// Minimal 2D vertex structure for polylines.
#[derive(Debug, Clone)]
pub struct PlineVertex2D {
    pub x: f64,
    pub y: f64,
    pub bulge: f64,
}

impl PlineVertex2D {
    pub fn new(x: f64, y: f64) -> Self {
        PlineVertex2D { x, y, bulge: 0.0 }
    }
}

/// Converts one of your 3D polygons (assumed to lie in the XY plane) into
/// a sequence of 2D polyline vertices. We set `bulge = 0.0` for each edge.
pub fn polygon_to_polyline2d<S: Clone>(poly: &Polygon<S>) -> Vec<PlineVertex2D> {
    let mut result = Vec::new();

    if poly.vertices.len() < 2 {
        return result; // degenerate or empty polygon
    }

    // We assume the polygon is already in the XY plane (z ~ 0).
    // If your polygons might have arcs, you'd need more logic to detect + store bulge, etc.
    for v in &poly.vertices {
        result.push(PlineVertex2D {
            x: v.pos.x, // x in plane
            y: v.pos.y, // y in plane
            bulge: 0.0, // bulge=0 => line segment
        });
    }

    result
}

/// Builds a CSG from a list of 2D polylines (in XY). We treat each polyline as one
/// flat polygon, lying in Z=0 with normal +Z. If you have multiple polylines,
/// they become multiple polygons in the returned CSG. For more advanced logic
/// (like holes, or combining them), you can union them or store them differently.
pub fn build_csg_from_polyline<S: Clone>(
    loops: &Vec<Vec<PlineVertex2D>>,
) -> CSG<S> {
    let mut all_polygons = Vec::new();

    // Normal for all polygons is +Z in a simple 2D XY scenario
    let plane_normal = nalgebra::Vector3::new(0.0, 0.0, 1.0);

    for loop_ in loops {
        if loop_.len() < 3 {
            // skip degenerate
            continue;
        }

        // Convert the loop of (x,y,bulge=0) into a single Polygon with flat shading
        let mut poly_verts = Vec::with_capacity(loop_.len());

        for v2d in loop_ {
            let pt = nalgebra::Point3::new(v2d.x, v2d.y, 0.0);
            poly_verts.push(Vertex::new(pt, plane_normal));
        }

        // Make one Polygon with an auto-computed plane
        let poly_3d = Polygon::new(poly_verts, None);
        all_polygons.push(poly_3d);
    }

    // Combine all polygons into a single CSG
    CSG::from_polygons(all_polygons)
}

/// Convert your 2D vertices into a cavalier_contours Polyline<f64>, making it closed.
pub fn polyline2d_to_cc_polyline(verts: &[PlineVertex2D]) -> Polyline<f64> {
    let mut pl = Polyline::with_capacity(verts.len(), true);
    for v in verts {
        pl.add(v.x, v.y, v.bulge);
    }
    pl
}

/// Computes the signed area of a closed 2D polyline via the shoelace formula.
/// We assume `pline.is_closed() == true` and it has at least 2 vertices.
/// Returns positive area if CCW, negative if CW. Near-zero => degenerate.
fn pline_area(pline: &Polyline<f64>) -> f64 {
    if pline.vertex_count() < 3 {
        return 0.0;
    }
    let mut area = 0.0;
    let n = pline.vertex_count();
    for i in 0..n {
        let j = (i + 1) % n;
        let (x_i, y_i) = (pline.at(i).x, pline.at(i).y);
        let (x_j, y_j) = (pline.at(j).x, pline.at(j).y);
        area += x_i * y_j - y_i * x_j;
    }
    0.5 * area
}

/// Build a new CSG from a set of 2D polylines in XY. Each polyline
/// is turned into one polygon at z=0. If a union produced multiple
/// loops, you will get multiple polygons in the final CSG.
pub fn build_csg_from_cc_polylines<S: Clone>(
    loops: Vec<Polyline<f64>>,
) -> CSG<S> {
    let mut all_polygons = Vec::new();
    let plane_normal = Vector3::new(0.0, 0.0, 1.0);

    for pl in loops {
        // Convert each Polyline into a single polygon in z=0.
        // If you need arcs, you could subdivide by bulge, etc. This example ignores arcs for simplicity.
        if pl.vertex_count() >= 3 {
            let mut poly_verts = Vec::with_capacity(pl.vertex_count());
            for i in 0..pl.vertex_count() {
                let v = pl.at(i);
                poly_verts.push(Vertex::new(
                    nalgebra::Point3::new(v.x, v.y, 0.0),
                    plane_normal
                ));
            }
            all_polygons.push(Polygon::new(poly_verts, None));
        }
    }

    CSG::from_polygons(all_polygons)
}

/// Flatten a `CSG` into the XY plane and union all polygons' outlines,
/// returning a new `CSG` that may contain multiple polygons (loops) if disjoint.
///
/// We skip "degenerate" loops whose area is near zero, both before
/// and after performing the union. This helps avoid collinear or
/// duplicate edges that can cause issues in `cavalier_contours`.
pub fn flatten_and_union<S: Clone>(csg: &CSG<S>) -> CSG<S> {
    let eps_area = 1e-9;
    let eps_pos = 1e-5;

    // 1) Convert each 3D polygon into a 2D polyline on z=0,
    //    remove duplicates, skip degenerate (zero-area) loops.
    let mut polylines_2d = Vec::new();
    for poly in &csg.polygons {
        let pline2d = polygon_to_polyline2d(poly);
        let cc_poly = polyline2d_to_cc_polyline(&pline2d);
        cc_poly.remove_redundant(eps_pos);

        // Check area; skip if below threshold
        if pline_area(&cc_poly).abs() > eps_area {
            polylines_2d.push(cc_poly);
        }
    }
    if polylines_2d.is_empty() {
        return CSG::new();
    }

    // 2) Repeatedly union all polylines. Because 2D union can yield multiple disjoint loops,
    //    we store them all, filtering out degenerate loops each time.
    let mut union_acc: Vec<Polyline<f64>> = vec![polylines_2d[0].clone()];

    let options = PlineBooleanOptions {
        pline1_aabb_index: None,
        pos_equal_eps: eps_pos,
        ..Default::default()
    };

    for next_pl in &polylines_2d[1..] {
        let mut new_acc = Vec::new();
        for loop_pl in &union_acc {
            let union_result = polyline_boolean(loop_pl, next_pl, BooleanOp::Or, &options);
            // union_result.pos_plines are the union loops in 2D
            for res_pl in union_result.pos_plines {
                let area = pline_area(&res_pl.pline).abs();
                if area > eps_area {
                    new_acc.push(res_pl.pline);
                }
            }
        }
        union_acc = new_acc;
        if union_acc.is_empty() {
            break; // everything degenerated
        }
    }

    // 3) Convert the final union loops back to polygons in Z=0.
    build_csg_from_cc_polylines(union_acc)
}

/// Given a normal vector `n`, build two perpendicular unit vectors `u` and `v` so that
/// {u, v, n} forms an orthonormal basis. `n` is assumed non‐zero.
fn build_orthonormal_basis(n: nalgebra::Vector3<f64>) -> (nalgebra::Vector3<f64>, nalgebra::Vector3<f64>) {
    // Normalize the given normal
    let n = n.normalize();

    // Pick a vector that is not parallel to `n`. For instance, pick the axis
    // which has the smallest absolute component in `n`, and cross from there.
    let mut other = nalgebra::Vector3::new(0.0, 0.0, 0.0);

    // We choose the axis with the smallest absolute value in n,
    // because crossing with that is least likely to cause numeric issues.
    if n.x.abs() < n.y.abs() && n.x.abs() < n.z.abs() {
        other.x = 1.0;
    } else if n.y.abs() < n.z.abs() {
        other.y = 1.0;
    } else {
        other.z = 1.0;
    }

    // v = n × other
    let v = n.cross(&other).normalize();
    // u = v × n
    let u = v.cross(&n).normalize();

    (u, v)
}

/// A vertex of a polygon, holding position and normal.
#[derive(Debug, Clone)]
pub struct Vertex {
    pub pos: Point3<f64>,
    pub normal: Vector3<f64>,
}

impl Vertex {
    pub fn new(pos: Point3<f64>, normal: Vector3<f64>) -> Self {
        Vertex { pos, normal }
    }

    /// Flip orientation-specific data (like normals)
    pub fn flip(&mut self) {
        self.normal = -self.normal;
    }

    /// Linearly interpolate between `self` and `other` by parameter `t`.
    pub fn interpolate(&self, other: &Vertex, t: f64) -> Vertex {
        // For positions (Point3): p(t) = p0 + t * (p1 - p0)
        let new_pos = self.pos + (other.pos - self.pos) * t;

        // For normals (Vector3): n(t) = n0 + t * (n1 - n0)
        let new_normal = self.normal + (other.normal - self.normal) * t;
        Vertex::new(new_pos, new_normal)
    }
}

/// A plane in 3D space defined by a normal and a w-value
#[derive(Debug, Clone)]
pub struct Plane {
    pub normal: Vector3<f64>,
    pub w: f64,
}

impl Plane {
    /// Create a plane from three points
    pub fn from_points(a: &Point3<f64>, b: &Point3<f64>, c: &Point3<f64>) -> Plane {
        let n = (b - a).cross(&(c - a)).normalize();
        if n.magnitude() < EPSILON {
            panic!("Degenerate polygon: vertices do not define a plane");
        }
        Plane {
            normal: n,
            w: n.dot(&a.coords),
        }
    }

    pub fn flip(&mut self) {
        self.normal = -self.normal;
        self.w = -self.w;
    }

    /// Split `polygon` by this plane if needed, distributing the results into
    /// `coplanar_front`, `coplanar_back`, `front`, and `back`.
    pub fn split_polygon<S: Clone>(
        &self,
        polygon: &Polygon<S>,
        coplanar_front: &mut Vec<Polygon<S>>,
        coplanar_back: &mut Vec<Polygon<S>>,
        front: &mut Vec<Polygon<S>>,
        back: &mut Vec<Polygon<S>>,
    ) {
        const COPLANAR: i32 = 0;
        const FRONT: i32 = 1;
        const BACK: i32 = 2;
        const SPANNING: i32 = 3;

        let mut polygon_type = 0;
        let mut types = Vec::with_capacity(polygon.vertices.len());

        // Classify each vertex
        for v in &polygon.vertices {
            let t = self.normal.dot(&v.pos.coords) - self.w;
            let vertex_type = if t < -EPSILON {
                BACK
            } else if t > EPSILON {
                FRONT
            } else {
                COPLANAR
            };
            polygon_type |= vertex_type;
            types.push(vertex_type);
        }

        match polygon_type {
            COPLANAR => {
                // Coincident normals => belongs in front vs. back
                if self.normal.dot(&polygon.plane.normal) > 0.0 {
                    coplanar_front.push(polygon.clone());
                } else {
                    coplanar_back.push(polygon.clone());
                }
            }
            FRONT => {
                front.push(polygon.clone());
            }
            BACK => {
                back.push(polygon.clone());
            }
            _ => {
                // SPANNING
                let mut f: Vec<Vertex> = Vec::new();
                let mut b: Vec<Vertex> = Vec::new();
                let vcount = polygon.vertices.len();

                for i in 0..vcount {
                    let j = (i + 1) % vcount;
                    let ti = types[i];
                    let tj = types[j];
                    let vi = &polygon.vertices[i];
                    let vj = &polygon.vertices[j];

                    if ti != BACK {
                        f.push(vi.clone());
                    }
                    if ti != FRONT {
                        b.push(vi.clone());
                    }

                    if (ti | tj) == SPANNING {
                        let denom = self.normal.dot(&(vj.pos - vi.pos));
                        // Avoid dividing by zero
                        if denom.abs() > EPSILON {
                            let t = (self.w - self.normal.dot(&vi.pos.coords)) / denom;
                            let v = vi.interpolate(vj, t);
                            f.push(v.clone());
                            b.push(v);
                        }
                    }
                }

                if f.len() >= 3 {
                    front.push(Polygon::new(f, polygon.shared.clone()));
                }
                if b.len() >= 3 {
                    back.push(Polygon::new(b, polygon.shared.clone()));
                }
            }
        }
    }
}

// Helper function to subdivide triangles
fn subdivide_triangle(tri: [Vertex; 3]) -> Vec<[Vertex; 3]> {
    let v0 = tri[0].clone();
    let v1 = tri[1].clone();
    let v2 = tri[2].clone();

    let v01 = v0.interpolate(&v1, 0.5);
    let v12 = v1.interpolate(&v2, 0.5);
    let v20 = v2.interpolate(&v0, 0.5);

    vec![
        [v0.clone(),  v01.clone(), v20.clone()],
        [v01.clone(), v1.clone(),  v12.clone()],
        [v20.clone(), v12.clone(), v2.clone()],
        [v01,         v12,         v20],
    ]
}

/// A convex polygon, defined by a list of vertices and a plane.
/// - `S` is the generic "shared" data type, stored as `Option<S>`.
#[derive(Debug, Clone)]
pub struct Polygon<S: Clone> {
    pub vertices: Vec<Vertex>,
    pub shared: Option<S>,
    pub plane: Plane,
}

impl<S: Clone> Polygon<S> {
    /// Create a polygon from vertices
    pub fn new(vertices: Vec<Vertex>, shared: Option<S>) -> Self {
        assert!(
            vertices.len() >= 3,
            "Polygon::new requires at least 3 vertices"
        );
    
        let plane = Plane::from_points(
            &vertices[0].pos,
            &vertices[1].pos,
            &vertices[2].pos,
        );
        Polygon { vertices, shared, plane }
    }

    pub fn flip(&mut self) {
        self.vertices.reverse();
        for v in &mut self.vertices {
            v.flip();
        }
        self.plane.flip();
    }

    /// Triangulate this polygon into a list of triangles, each triangle is [v0, v1, v2].
    pub fn triangulate(&self) -> Vec<[Vertex; 3]> {
        let mut triangles = Vec::new();
        if self.vertices.len() < 3 {
            return triangles;
        }
        let v0 = self.vertices[0].clone();
        for i in 1..(self.vertices.len() - 1) {
            triangles.push([
                v0.clone(),
                self.vertices[i].clone(),
                self.vertices[i + 1].clone(),
            ]);
        }
        triangles
    }

    /// Subdivide this polygon into smaller triangles.
    /// Returns a list of new triangles (each is a [Vertex; 3]).
    pub fn subdivide_triangles(&self, levels: u32) -> Vec<[Vertex; 3]> {
        // 1) Triangulate the polygon as it is.
        let base_tris = self.triangulate();

        // 2) For each triangle, subdivide 'levels' times.
        let mut result = Vec::new();
        for tri in base_tris {
            // We'll keep a queue of triangles to process
            let mut queue = vec![tri];
            for _ in 0..levels {
                let mut next_level = Vec::new();
                for t in queue {
                    let subs = subdivide_triangle(t);
                    next_level.extend(subs);
                }
                queue = next_level;
            }
            result.extend(queue);
        }

        result
    }

    /// Recompute this polygon's plane from the first 3 vertices,
    /// then set all vertices' normals to match that plane (flat shading).
    pub fn recalc_plane_and_normals(&mut self) {
        if self.vertices.len() < 3 {
            return; // degenerate or empty
        }
        // Recompute the plane from the first 3 vertices
        self.plane = Plane::from_points(
            &self.vertices[0].pos,
            &self.vertices[1].pos,
            &self.vertices[2].pos,
        );

        // Assign each vertex’s normal to match the plane
        let new_normal = self.plane.normal;
        for v in &mut self.vertices {
            v.normal = new_normal;
        }
    }

    // --- New getters/setters for shared data ---

    /// Returns a reference to the shared data, if any.
    pub fn shared_data(&self) -> Option<&S> {
        self.shared.as_ref()
    }

    /// Returns a mutable reference to the shared data, if any.
    pub fn shared_data_mut(&mut self) -> Option<&mut S> {
        self.shared.as_mut()
    }

    /// Sets the shared data to the given value.
    pub fn set_shared_data(&mut self, data: S) {
        self.shared = Some(data);
    }
}

/// A BSP tree node, containing polygons plus optional front/back subtrees
#[derive(Debug, Clone)]
pub struct Node<S: Clone> {
    pub plane: Option<Plane>,
    pub front: Option<Box<Node<S>>>,
    pub back: Option<Box<Node<S>>>,
    pub polygons: Vec<Polygon<S>>,
}

impl<S: Clone> Node<S> {
    pub fn new(polygons: Vec<Polygon<S>>) -> Self {
        let mut node = Node {
            plane: None,
            front: None,
            back: None,
            polygons: Vec::new(),
        };
        if !polygons.is_empty() {
            node.build(&polygons);
        }
        node
    }

    /// Invert all polygons in the BSP tree
    pub fn invert(&mut self) {
        for p in &mut self.polygons {
            p.flip();
        }
        if let Some(ref mut plane) = self.plane {
            plane.flip();
        }
        if let Some(ref mut front) = self.front {
            front.invert();
        }
        if let Some(ref mut back) = self.back {
            back.invert();
        }
        std::mem::swap(&mut self.front, &mut self.back);
    }

    /// Recursively remove all polygons in `polygons` that are inside this BSP tree
    pub fn clip_polygons(&self, polygons: &[Polygon<S>]) -> Vec<Polygon<S>> {
        if self.plane.is_none() {
            return polygons.to_vec();
        }

        let plane = self.plane.as_ref().unwrap();
        let mut front: Vec<Polygon<S>> = Vec::new();
        let mut back: Vec<Polygon<S>> = Vec::new();

        for poly in polygons {
            plane.split_polygon(
                poly,
                &mut Vec::new(), // coplanar_front
                &mut Vec::new(), // coplanar_back
                &mut front,
                &mut back,
            );
        }

        if let Some(ref f) = self.front {
            front = f.clip_polygons(&front);
        }
        if let Some(ref b) = self.back {
            back = b.clip_polygons(&back);
        } else {
            back.clear();
        }

        front.extend(back);
        front
    }

    /// Remove all polygons in this BSP tree that are inside the other BSP tree
    pub fn clip_to(&mut self, bsp: &Node<S>) {
        self.polygons = bsp.clip_polygons(&self.polygons);
        if let Some(ref mut front) = self.front {
            front.clip_to(bsp);
        }
        if let Some(ref mut back) = self.back {
            back.clip_to(bsp);
        }
    }

    /// Return all polygons in this BSP tree
    pub fn all_polygons(&self) -> Vec<Polygon<S>> {
        let mut result = self.polygons.clone();
        if let Some(ref front) = self.front {
            result.extend(front.all_polygons());
        }
        if let Some(ref back) = self.back {
            result.extend(back.all_polygons());
        }
        result
    }

    /// Build a BSP tree from the given polygons
    pub fn build(&mut self, polygons: &[Polygon<S>]) {
        if polygons.is_empty() {
            return;
        }

        if self.plane.is_none() {
            self.plane = Some(polygons[0].plane.clone());
        }
        let plane = self.plane.clone().unwrap();

        let mut front: Vec<Polygon<S>> = Vec::new();
        let mut back: Vec<Polygon<S>> = Vec::new();

        for p in polygons {
            let mut coplanar_front = Vec::new();
            let mut coplanar_back = Vec::new();

            plane.split_polygon(
                p,
                &mut coplanar_front,
                &mut coplanar_back,
                &mut front,
                &mut back,
            );

            self.polygons.append(&mut coplanar_front);
            self.polygons.append(&mut coplanar_back);
        }

        if !front.is_empty() {
            if self.front.is_none() {
                self.front = Some(Box::new(Node::new(vec![])));
            }
            self.front.as_mut().unwrap().build(&front);
        }

        if !back.is_empty() {
            if self.back.is_none() {
                self.back = Some(Box::new(Node::new(vec![])));
            }
            self.back.as_mut().unwrap().build(&back);
        }
    }
}

/// The main CSG solid structure. Contains a list of polygons.
#[derive(Debug, Clone)]
pub struct CSG<S: Clone> {
    pub polygons: Vec<Polygon<S>>,
}

impl<S: Clone> CSG<S> {
    /// Create an empty CSG
    pub fn new() -> Self {
        CSG { polygons: Vec::new() }
    }

    /// Build a CSG from an existing polygon list
    pub fn from_polygons(polygons: Vec<Polygon<S>>) -> Self {
        let mut csg = CSG::new();
        csg.polygons = polygons;
        csg
    }

    /// Return the internal polygons
    pub fn to_polygons(&self) -> &[Polygon<S>] {
        &self.polygons
    }

    /// CSG union: this ∪ other
    pub fn union(&self, other: &CSG<S>) -> CSG<S> {
        let mut a = Node::new(self.polygons.clone());
        let mut b = Node::new(other.polygons.clone());

        a.clip_to(&b);
        b.clip_to(&a);
        b.invert();
        b.clip_to(&a);
        b.invert();
        a.build(&b.all_polygons());

        CSG::from_polygons(a.all_polygons())
    }

    /// CSG subtract: this \ other
    pub fn subtract(&self, other: &CSG<S>) -> CSG<S> {
        let mut a = Node::new(self.polygons.clone());
        let mut b = Node::new(other.polygons.clone());

        a.invert();
        a.clip_to(&b);
        b.clip_to(&a);
        b.invert();
        b.clip_to(&a);
        b.invert();
        a.build(&b.all_polygons());
        a.invert();

        CSG::from_polygons(a.all_polygons())
    }

    /// CSG intersect: this ∩ other
    pub fn intersect(&self, other: &CSG<S>) -> CSG<S> {
        let mut a = Node::new(self.polygons.clone());
        let mut b = Node::new(other.polygons.clone());

        a.invert();
        b.clip_to(&a);
        b.invert();
        a.clip_to(&b);
        b.clip_to(&a);
        a.build(&b.all_polygons());
        a.invert();

        CSG::from_polygons(a.all_polygons())
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
    /// - `size`: the width and height of the square (default [1.0, 1.0])
    /// - `center`: if `true`, center the square about (0,0); otherwise bottom-left is at (0,0).
    ///
    /// # Example
    /// let sq = CSG::square(None);
    /// // or with custom params:
    /// let sq2 = CSG::square(Some(([2.0, 3.0], true)));
    pub fn square(params: Option<([f64; 2], bool)>) -> CSG<S> {
        let (size, center) = match params {
            Some((sz, c)) => (sz, c),
            None => ([1.0, 1.0], false),
        };

        let (w, h) = (size[0], size[1]);
        let (x0, y0, x1, y1) = if center {
            (-w / 2.0, -h / 2.0, w / 2.0, h / 2.0)
        } else {
            (0.0, 0.0, w, h)
        };

        // Single 2D polygon, normal = +Z
        let normal = Vector3::new(0.0, 0.0, 1.0);
        let vertices = vec![
            Vertex::new(Point3::new(x0, y0, 0.0), normal),
            Vertex::new(Point3::new(x1, y0, 0.0), normal),
            Vertex::new(Point3::new(x1, y1, 0.0), normal),
            Vertex::new(Point3::new(x0, y1, 0.0), normal),
        ];
        CSG::from_polygons(vec![Polygon::new(vertices, None)])
    }

    /// Creates a 2D circle in the XY plane.
    pub fn circle(params: Option<(f64, usize)>) -> CSG<S> {
        let (r, segments) = match params {
            Some((radius, segs)) => (radius, segs),
            None => (1.0, 32),
        };

        let mut verts = Vec::with_capacity(segments);
        let normal = Vector3::new(0.0, 0.0, 1.0);

        for i in 0..segments {
            let theta = 2.0 * std::f64::consts::PI * (i as f64) / (segments as f64);
            let x = r * theta.cos();
            let y = r * theta.sin();
            verts.push(Vertex::new(Point3::new(x, y, 0.0), normal));
        }

        CSG::from_polygons(vec![Polygon::new(verts, None)])
    }

    /// Creates a 2D polygon in the XY plane from a list of `[x, y]` points.
    ///
    /// # Parameters
    ///
    /// - `points`: a sequence of 2D points (e.g. `[[0.0,0.0], [1.0,0.0], [0.5,1.0]]`)
    ///   describing the polygon boundary in order.
    ///
    /// **Note:** This simple version ignores 'paths' and holes. For more complex
    /// polygons, we'll have to handle multiple paths, winding order, holes, etc.
    ///
    /// # Example
    /// let pts = vec![[0.0, 0.0], [2.0, 0.0], [1.0, 1.5]];
    /// let poly2d = CSG::polygon_2d(&pts);
    pub fn polygon_2d(points: &[[f64; 2]]) -> CSG<S> {
        assert!(points.len() >= 3, "polygon_2d requires at least 3 points");
        let normal = Vector3::new(0.0, 0.0, 1.0);
        let mut verts = Vec::with_capacity(points.len());
        for p in points {
            verts.push(Vertex::new(Point3::new(p[0], p[1], 0.0), normal));
        }
        CSG::from_polygons(vec![Polygon::new(verts, None)])
    }

    /// Construct an axis-aligned cube by creating a 2D square and then
    /// extruding it between two parallel planes (using `extrude_between`).
    /// We translate the final shape so that its center is at `center`,
    /// and each axis extends ± its corresponding `radius`.
    pub fn cube(options: Option<(&[f64; 3], &[f64; 3])>) -> CSG<S> {
        let (center, radius) = match options {
            Some((c, r)) => (*c, *r),
            None => ([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
        };
        let c = Vector3::new(center[0], center[1], center[2]);
        let r = Vector3::new(radius[0], radius[1], radius[2]);
    
        // 1) Create a centered square in the XY plane of width 2*r.x and height 2*r.y.
        let base_2d = CSG::square(Some(([2.0 * r.x, 2.0 * r.y], true)));
        // Our 2D shape has exactly one polygon:
        let bottom_poly = &base_2d.polygons[0];
    
        // 2) Make the top polygon by translating this bottom polygon in +Z by 2*r.z.
        let top_poly_csg = CSG::from_polygons(vec![bottom_poly.clone()])
            .translate(Vector3::new(0.0, 0.0, 2.0 * r.z));
        let top_poly = &top_poly_csg.polygons[0];
    
        // 3) Extrude between them to get a rectangular prism.
        let prism = CSG::extrude_between(bottom_poly, top_poly, true);
    
        // 4) Finally, shift so that the prism is centered at `c` in Z:
        //    originally it spans z from 0..2*r.z; translating by -r.z moves it to -r.z..+r.z,
        //    then we shift by c. This yields the final bounding box:
        //       [c.x - r.x .. c.x + r.x,
        //        c.y - r.y .. c.y + r.y,
        //        c.z - r.z .. c.z + r.z].
        prism
            .translate(Vector3::new(0.0, 0.0, -r.z))
            .translate(c)
    }

    /// Construct a sphere with optional center, radius, slices, stacks
    pub fn sphere(options: Option<(&[f64; 3], f64, usize, usize)>) -> CSG<S> {
        let (center, radius, slices, stacks) = match options {
            Some((c, r, sl, st)) => (*c, r, sl, st),
            None => ([0.0, 0.0, 0.0], 1.0, 16, 8),
        };

        let c = Vector3::new(center[0], center[1], center[2]);
        let mut polygons = Vec::new();

        for i in 0..slices {
            for j in 0..stacks {
                let mut vertices = Vec::new();

                let vertex = |theta: f64, phi: f64| {
                    let dir = Vector3::new(
                        theta.cos() * phi.sin(),
                        phi.cos(),
                        theta.sin() * phi.sin(),
                    );
                    Vertex::new(Point3::new(c.x + dir.x * radius,
                                            c.y + dir.y * radius,
                                            c.z + dir.z * radius),
                                dir)
                };

                let t0 = i as f64 / slices as f64;
                let t1 = (i + 1) as f64 / slices as f64;
                let p0 = j as f64 / stacks as f64;
                let p1 = (j + 1) as f64 / stacks as f64;

                let theta0 = t0 * std::f64::consts::TAU;
                let theta1 = t1 * std::f64::consts::TAU;
                let phi0 = p0 * std::f64::consts::PI;
                let phi1 = p1 * std::f64::consts::PI;

                vertices.push(vertex(theta0, phi0));
                if j > 0 {
                    vertices.push(vertex(theta1, phi0));
                }
                if j < stacks - 1 {
                    vertices.push(vertex(theta1, phi1));
                }
                vertices.push(vertex(theta0, phi1));

                polygons.push(Polygon::new(vertices, None));
            }
        }

        CSG::from_polygons(polygons)
    }

    /// Construct a cylinder whose centerline goes from `start` to `end`,
    /// with a circular cross-section of given `radius`. We first build a
    /// 2D circle in the XY plane, then extrude two copies of it in the Z
    /// direction (via `extrude_between`), and finally rotate/translate
    /// so that the cylinder aligns with `start -> end`.
    pub fn cylinder(options: Option<(&[f64; 3], &[f64; 3], f64, usize)>) -> CSG<S> {
        let (start, end, radius, slices) = match options {
            Some((s, e, r, sl)) => (*s, *e, r, sl),
            None => ([0.0, -1.0, 0.0], [0.0, 1.0, 0.0], 1.0, 16),
        };
    
        let s = Vector3::new(start[0], start[1], start[2]);
        let e = Vector3::new(end[0], end[1], end[2]);
        let ray = e - s;
        let length = ray.norm();
    
        // 1) Build a 2D circle in the XY plane of radius `radius`.
        let circle_2d = CSG::circle(Some((radius, slices)));
        let bottom_poly = &circle_2d.polygons[0];
    
        // 2) Translate a copy of that circle up in Z by `length` to form the top.
        let top_poly_csg = CSG::from_polygons(vec![bottom_poly.clone()])
            .translate(Vector3::new(0.0, 0.0, length));
        let top_poly = &top_poly_csg.polygons[0];
    
        // 3) Extrude between the bottom and top polygons to form a basic cylinder along the Z-axis.
        let raw_cyl = CSG::extrude_between(bottom_poly, top_poly, true);
    
        // 4) Orient the cylinder so that its axis goes from `start` to `end`.
        //    We rotate from +Z to `ray` using an axis-angle rotation.
        if length.abs() < f64::EPSILON {
            // Degenerate case: start == end, just return the shape without orientation.
            return raw_cyl.translate(s);
        }
    
        let z_unit = Vector3::new(0.0, 0.0, 1.0);
        let direction = ray / length;
        let axis = z_unit.cross(&direction);
        let angle = z_unit.angle(&direction);
    
        let rotation = if axis.magnitude() < f64::EPSILON {
            // If `ray` is parallel or anti-parallel to Z, handle that separately:
            // e.g. if ray is in +Z direction, no rotation needed; if in -Z, rotate 180° about X or Y.
            if z_unit.dot(&direction) > 0.0 {
                Matrix4::identity()
            } else {
                // 180° flip around X-axis (or Y-axis), for a perfect reversal.
                Rotation3::from_axis_angle(&Vector3::x_axis(), std::f64::consts::PI).to_homogeneous()
            }
        } else {
            let axis_norm = axis.normalize();
            Rotation3::from_axis_angle(&nalgebra::Unit::new_unchecked(axis_norm), angle).to_homogeneous()
        };
    
        // 5) Apply the rotation, then translate so the bottom sits at `start`.
        raw_cyl
            .transform(&rotation)
            .translate(s)
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
    /// let csg_poly = CSG::polyhedron(pts, &fcs);
    /// ```
    pub fn polyhedron(points: &[[f64; 3]], faces: &[Vec<usize>]) -> CSG<S> {
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
                    panic!("Face index {} is out of range (points.len = {}).", idx, points.len());
                }
                let [x, y, z] = points[idx];
                face_vertices.push(Vertex::new(
                    Point3::new(x, y, z),
                    Vector3::zeros(), // we'll set this later
                ));
            }

            // Build the polygon (plane is auto-computed from first 3 vertices).
            let mut poly = Polygon::new(face_vertices, None);

            // Optionally, set each vertex normal to match the polygon’s plane normal,
            // so that shading in many 3D viewers looks correct.
            let plane_normal = poly.plane.normal;
            for v in &mut poly.vertices {
                v.normal = plane_normal;
            }
            polygons.push(poly);
        }

        CSG::from_polygons(polygons)
    }

    /// Transform all vertices in this CSG by a given 4×4 matrix.
    pub fn transform(&self, mat: &Matrix4<f64>) -> CSG<S> {
        let mat_inv_transpose = mat.try_inverse().unwrap().transpose();
        let mut csg = self.clone();

        for poly in &mut csg.polygons {
            for vert in &mut poly.vertices {
                // Position
                let hom_pos = mat * vert.pos.to_homogeneous();
                vert.pos = Point3::from_homogeneous(hom_pos).unwrap();
    
                // Normal
                vert.normal = mat_inv_transpose.transform_vector(&vert.normal).normalize();
            }
    
            // Plane normal
            poly.plane.normal = mat_inv_transpose.transform_vector(&poly.plane.normal).normalize();
    
            // Plane w
            if let Some(first_vert) = poly.vertices.get(0) {
                poly.plane.w = poly.plane.normal.dot(&first_vert.pos.coords);
            }
        }

        csg
    }

    pub fn translate(&self, v: Vector3<f64>) -> CSG<S> {
        let translation = Translation3::from(v);
        // Convert to a Matrix4
        let mat4 = translation.to_homogeneous();
        self.transform(&mat4)
    }

    pub fn rotate(&self, x_deg: f64, y_deg: f64, z_deg: f64) -> CSG<S> {
        let rx = Rotation3::from_axis_angle(&Vector3::x_axis(), x_deg.to_radians());
        let ry = Rotation3::from_axis_angle(&Vector3::y_axis(), y_deg.to_radians());
        let rz = Rotation3::from_axis_angle(&Vector3::z_axis(), z_deg.to_radians());
        
        // Compose them in the desired order
        let rot = rz * ry * rx;
        self.transform(&rot.to_homogeneous())
    }

    pub fn scale(&self, sx: f64, sy: f64, sz: f64) -> CSG<S> {
        let mat4 = Matrix4::new_nonuniform_scaling(&Vector3::new(sx, sy, sz));
        self.transform(&mat4)
    }

    /// Mirror across X=0, Y=0, or Z=0 plane
    pub fn mirror(&self, axis: Axis) -> CSG<S> {
        let (sx, sy, sz) = match axis {
            Axis::X => (-1.0, 1.0, 1.0),
            Axis::Y => (1.0, -1.0, 1.0),
            Axis::Z => (1.0, 1.0, -1.0),
        };

        // We can just use a "non-uniform scaling" matrix that
        // flips exactly one axis:
        let mat = Matrix4::new_nonuniform_scaling(&Vector3::new(sx, sy, sz));
        self.transform(&mat)
    }

    /// Compute the convex hull of all vertices in this CSG.
    pub fn convex_hull(&self) -> CSG<S> {
	// Gather all (x, y, z) coordinates from the polygons
        let points: Vec<Vec<f64>> = self
            .polygons
            .iter()
            .flat_map(|poly| {
                poly.vertices.iter().map(|v| vec![v.pos.x, v.pos.y, v.pos.z])
            })
            .collect();

        // Compute convex hull using the robust wrapper
        let hull = ConvexHullWrapper::try_new(&points, None)
            .expect("Failed to compute convex hull");

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

        CSG::from_polygons(polygons)
    }

    /// Compute the Minkowski sum: self ⊕ other
    ///
    /// Naive approach: Take every vertex in `self`, add it to every vertex in `other`,
    /// then compute the convex hull of all resulting points.
    pub fn minkowski_sum(&self, other: &CSG<S>) -> CSG<S> {
        // Collect all vertices (x, y, z) from self
        let verts_a: Vec<Point3<f64>> = self.polygons
            .iter()
            .flat_map(|poly| poly.vertices.iter().map(|v| v.pos))
            .collect();

        // Collect all vertices from other
        let verts_b: Vec<Point3<f64>> = other.polygons
            .iter()
            .flat_map(|poly| poly.vertices.iter().map(|v| v.pos))
            .collect();

        // For Minkowski, add every point in A to every point in B
        let mut sum_points = Vec::with_capacity(verts_a.len() * verts_b.len());
        for a in &verts_a {
            for b in &verts_b {
                sum_points.push(vec![a.x + b.x, a.y + b.y, a.z + b.z]);
            }
        }

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

        CSG::from_polygons(polygons)
    }

    /// Subdivide all polygons in this CSG 'levels' times, returning a new CSG.
    /// This results in a triangular mesh with more detail.
    pub fn subdivide_triangles(&self, levels: u32) -> CSG<S> {
        if levels == 0 {
            return self.clone();
        }

        let mut new_polygons = Vec::new();
        for poly in &self.polygons {
            // Subdivide the polygon into many smaller triangles
            let sub_tris = poly.subdivide_triangles(levels);
            // Convert each small tri back into a Polygon with 3 vertices
            // (you can keep the same 'shared' data or None).
            for tri in sub_tris {
                new_polygons.push(
                    Polygon::new(vec![tri[0].clone(), tri[1].clone(), tri[2].clone()], poly.shared.clone())
                );
            }
        }

        CSG::from_polygons(new_polygons)
    }

    /// Renormalize all polygons in this CSG by re-computing each polygon’s plane
    /// and assigning that plane’s normal to all vertices.
    pub fn renormalize(&mut self) {
        for poly in &mut self.polygons {
            poly.recalc_plane_and_normals();
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
    /// A `Vec` of `(Point3<f64>, f64)` where:
    /// - `Point3<f64>` is the intersection coordinate in 3D,
    /// - `f64` is the distance (the ray parameter t) from `origin`.
    pub fn ray_intersections(
        &self,
        origin: &Point3<f64>,
        direction: &Vector3<f64>,
    ) -> Vec<(Point3<f64>, f64)> {
        let ray = Ray::new(*origin, *direction);
        let iso = Isometry3::identity(); // No transformation on the triangles themselves.

        let mut hits = Vec::new();

        // 1) For each polygon in the CSG:
        for poly in &self.polygons {
            // 2) Triangulate it if necessary:
            let triangles = poly.triangulate();

            // 3) For each triangle, do a ray–triangle intersection test:
            for tri in triangles {
                let a = tri[0].pos;
                let b = tri[1].pos;
                let c = tri[2].pos;

                // Construct a parry Triangle shape from the 3 vertices:
                let triangle = Triangle::new(a, b, c);

                // Ray-cast against the triangle:
                if let Some(hit) = triangle.cast_ray_and_get_normal(&iso, &ray, f64::MAX, true) {
                    let point_on_ray = ray.point_at(hit.time_of_impact);
                    hits.push((
                        Point3::from(point_on_ray.coords),
                        hit.time_of_impact,
                    ));
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
    /// This is similar to OpenSCAD's `linear_extrude(height=...)` assuming
    /// the base 2D shape is in the XY plane with a +Z normal.
    pub fn extrude(&self, height: f64) -> CSG<S> {
        // Simply call the new extrude_along method with a (0,0,height) vector
        self.extrude_along(Vector3::new(0.0, 0.0, height))
    }
    
    /// Linearly extrude this (2D) shape along an arbitrary 3D direction vector.
    ///
    /// - The shape is “swept” from its original location to a translated “top” copy
    ///   offset by `direction`.
    /// - Side walls are formed between the bottom and top edges.
    /// - The shape is assumed to be “2D” in the sense that each polygon typically
    ///   lies in a single plane (e.g. XY). For best results, your polygons’ normals
    ///   should be consistent.
    pub fn extrude_along(&self, direction: Vector3<f64>) -> CSG<S> {
        // Collect all new polygons here
        let mut new_polygons = Vec::new();
    
        // 1) Bottom polygons = original polygons
        //    (assuming they are in some plane, e.g. XY). We just clone them.
        for poly in &self.polygons {
            new_polygons.push(poly.clone());
        }
    
        // 2) Top polygons = translate each original polygon by `direction`.
        //    The orientation of their normals will remain the same unless you decide to flip them.
        let top_polygons = self.translate(direction).polygons;
        new_polygons.extend(top_polygons.iter().cloned());
    
        // 3) Side polygons = For each polygon in `self`, connect its edges
        //    from the original to the corresponding edges in the translated version.
        //
        //    We'll iterate over each polygon’s vertices. For each edge (v[i], v[i+1]),
        //    we form a rectangular side quad with (v[i]+direction, v[i+1]+direction).
        //    That is, a quad [b_i, b_j, t_j, t_i].
        let bottom_polys = &self.polygons;
        let top_polys = &top_polygons;
    
        for (poly_bottom, poly_top) in bottom_polys.iter().zip(top_polys.iter()) {
            let vcount = poly_bottom.vertices.len();
            if vcount < 3 {
                continue; // skip degenerate or empty polygons
            }
            for i in 0..vcount {
                let j = (i + 1) % vcount; // next index, wrapping around
                let b_i = &poly_bottom.vertices[i];
                let b_j = &poly_bottom.vertices[j];
                let t_i = &poly_top.vertices[i];
                let t_j = &poly_top.vertices[j];
    
                // Build a side quad [b_i, b_j, t_j, t_i].
                // Then push it as a new polygon.
                let side_poly = Polygon::new(
                    vec![b_i.clone(), b_j.clone(), t_j.clone(), t_i.clone()],
                    None
                );
                new_polygons.push(side_poly);
            }
        }
    
        // Combine into a new CSG
        CSG::from_polygons(new_polygons)
    }
    
    /// Extrudes (or "lofts") a closed 3D volume between two polygons in space.
    /// - `bottom` and `top` each have the same number of vertices `n`, in matching order.
    /// - Returns a new CSG whose faces are:
    ///   - The `bottom` polygon,
    ///   - The `top` polygon,
    ///   - `n` rectangular side polygons bridging each edge of `bottom` to the corresponding edge of `top`.
    pub fn extrude_between(bottom: &Polygon<S>, top: &Polygon<S>, flip_bottom_polygon: bool) -> CSG<S> {
        let n = bottom.vertices.len();
        assert_eq!(
            n,
            top.vertices.len(),
            "extrude_between: both polygons must have the same number of vertices"
        );
    
        // Conditionally flip the bottom polygon if requested.
        let bottom_poly = if flip_bottom_polygon {
            let mut flipped = bottom.clone();
            flipped.flip();
            flipped
        } else {
            bottom.clone()
        };
    
        // 1) Gather polygons: bottom + top
        //    (Depending on the orientation, you might want to flip one of them.)

        let mut polygons = vec![bottom_poly.clone(), top.clone()];
    
        // 2) For each edge (i -> i+1) in bottom, connect to the corresponding edge in top.
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
                None, // or carry over some shared data if you wish
            );
            polygons.push(side_poly);
        }
    
        CSG::from_polygons(polygons)
    }

    /// Rotate-extrude (revolve) this 2D shape around the Z-axis from 0..`angle_degs`
    /// by replicating the original polygon(s) at each step and calling `extrude_between`.
    /// Caps are added automatically if the revolve is partial (angle < 360°).
    pub fn rotate_extrude(&self, angle_degs: f64, segments: usize) -> CSG<S> {
        let angle_radians = angle_degs.to_radians();
        if segments < 2 {
            panic!("rotate_extrude requires at least 2 segments");
        }
    
        // We'll consider the revolve "closed" if the angle is effectively 360°
        let closed = (angle_degs - 360.0).abs() < EPSILON;
    
        // Collect all newly formed polygons here
        let mut result_polygons = Vec::new();
    
        // For each polygon in our original 2D shape:
        for original_poly in &self.polygons {
            let n_verts = original_poly.vertices.len();
            if n_verts < 3 {
                // Skip degenerate or empty polygons
                continue;
            }
    
            // 1) Create a list of rotated copies ("slices") of `original_poly`.
            //    We'll generate `segments+1` slices if it's a partial revolve,
            //    so that slices[0] = 0° and slices[segments] = angle_degs,
            //    giving us "segments" intervals to extrude_between.
            //    If `angle_degs == 360`, slices[segments] ends up co-located with slices[0].
            let mut slices = Vec::with_capacity(segments + 1);
            for i in 0..=segments {
                let frac = i as f64 / segments as f64;
                let theta = frac * angle_radians;
    
                // Build a rotation around Z by `theta`
                let rot = Rotation3::from_axis_angle(&Vector3::z_axis(), theta).to_homogeneous();
    
                // Transform this single polygon by that rotation
                let rotated_poly = CSG::from_polygons(vec![original_poly.clone()])
                    .transform(&rot)
                    .polygons[0]
                    .clone();
                slices.push(rotated_poly);
            }
    
            // 2) "Loft" between successive slices using `extrude_between`.
            //    - If it's a full 360 revolve, we do 0..(segments) and wrap around 
            //      from slices[segments-1] => slices[0].
            //    - If it's partial, we just do 0..(segments), which covers 
            //      slices[i] -> slices[i+1] for i=0..(segments-1).
            if closed {
                // Full revolve (angle ~ 360)
                for i in 0..(segments) {
                    // extrude_between slices[i] and slices[i+1], but on the last iteration
                    // slices[segments] is effectively the same as slices[0].
                    let bottom = &slices[i];
                    let top = &slices[(i + 1) % slices.len()]; // wraps around
                    let mut side_solid = CSG::extrude_between(bottom, top, false).polygons;
                    result_polygons.append(&mut side_solid);
                }
                // No separate caps needed for a closed revolve; it's already sealed.
            } else {
                // Partial revolve (angle < 360)
                // We form `segments` extrusions from i..i+1 (i=0..segments-1)
                for i in 0..segments {
                    let bottom = &slices[i];
                    let top = &slices[i + 1];
                    let mut side_solid = CSG::extrude_between(bottom, top, false).polygons;
                    result_polygons.append(&mut side_solid);
                }
    
                // 3) Add "caps" for the first and last slices so the revolve is closed at each end.
                //    Typically you want them oriented outward, so we can flip one cap if desired.
                let mut start_cap = slices[0].clone();
                // Flip the start cap so its normal faces outward instead of inward:
                start_cap.flip();
    
                // The end cap is slices[segments] as-is:
                let end_cap = slices[segments].clone();
    
                result_polygons.push(start_cap);
                result_polygons.push(end_cap);
            }
        }
    
        // Gather everything into a new CSG
        CSG::from_polygons(result_polygons)
    }

    /// Returns a `parry3d::bounding_volume::Aabb`.
    pub fn bounding_box(&self) -> Aabb {
        // Gather all points from all polygons.
        // parry expects a slice of `&Point3<f64>` or a slice of `na::Point3<f64>`.
        let mut all_points = Vec::new();
        for poly in &self.polygons {
            for v in &poly.vertices {
                all_points.push(v.pos);
            }
        }

        // If empty, return a degenerate AABB at origin or handle accordingly
        if all_points.is_empty() {
            return Aabb::new_invalid(); // or AABB::new(Point3::origin(), Point3::origin());
        }

        // Construct the parry AABB from points
        Aabb::from_points(&all_points)
    }

    /// Helper to collect all vertices from the CSG.
    pub fn vertices(&self) -> Vec<Vertex> {
        self.polygons
            .iter()
            .flat_map(|p| p.vertices.clone())
            .collect()
    }

    /// Approximate growing (outward offset) of the shape by a given distance (3D).
    /// This method unions translated copies of the shape along a sphere.
    pub fn grow(&self, distance: f64) -> CSG<S> {
        let resolution = 32;
        let sphere: CSG<S> = CSG::sphere(Some((&[0.0, 0.0, 0.0], distance, resolution, resolution / 2)));
        let sphere_vertices = sphere.vertices();
        let mut result = CSG::new();

        // Union the shape translated by each vertex of the sphere.
        for v in sphere_vertices {
            result = result.union(&self.translate(v.pos.coords));
        }
        result
    }

    /// Approximate shrinking (inward offset) of the shape by a given distance (3D).
    /// This method unions translated copies of the complement of the shape along a sphere,
    /// then inverts the result.
    pub fn shrink(&self, distance: f64) -> CSG<S> {
        let resolution = 32;
        let sphere: CSG<S> = CSG::sphere(Some((&[0.0, 0.0, 0.0], distance, resolution, resolution / 2)));
        let sphere_vertices = sphere.vertices();
        let complement = self.inverse();
        let mut result = CSG::new();

        for v in sphere_vertices {
            result = result.union(&complement.translate(v.pos.coords));
        }
        result.inverse()
    }

    /// Grows/outsets all polygons in the XY plane by `distance` using cavalier_contours parallel_offset.
    pub fn offset_2d(&self, distance: f64) -> CSG<S> {
        // 1) Convert each polygon to a 2D polyline in XY
        let mut all_plines_2d = Vec::new();
        for poly in &self.polygons {
            // Assume `poly` is in XY plane, gather the points:
            let pline2d = polygon_to_polyline2d(poly);
            all_plines_2d.push(pline2d);
        }

        // 2) For each polyline, build a cavalier_contours Polyline<f64>, offset it
        let mut offset_loops = Vec::new(); // each "loop" is a Vec<PlineVertex2D> in your 2D format

        for pline2d in all_plines_2d {
            // Convert to cavalier_contours Polyline (closed by default):
            let mut cpoly = Polyline::<f64>::with_capacity(pline2d.len(), /*is_closed=*/true);
            for v in &pline2d {
                cpoly.add(v.x, v.y, v.bulge);
            }

            // Remove any degenerate or redundant vertices:
            cpoly.remove_redundant(1e-5);

            // Perform the actual offset:
            let result_plines = cpoly.parallel_offset(-distance);

            // Convert the result polylines back into your PlineVertex2D
            for off_pl in result_plines {
                let mut out_loop = Vec::new();
                // The offset polyline is typically closed; iterate over all its vertices:
                for i in 0..off_pl.vertex_count() {
                    let vv = off_pl.at(i);
                    out_loop.push(PlineVertex2D {
                        x: vv.x,
                        y: vv.y,
                        bulge: vv.bulge,
                    });
                }
                // Collect this loop
                offset_loops.push(out_loop);
            }
        }

        // 3) Build a new CSG from those offset loops in XY:
        build_csg_from_polyline::<S>(&offset_loops)
    }

    /// Flattens (projects) this CSG onto the XY plane.
    /// Returns a new CSG containing only the resulting 2D polygons in z=0.
    pub fn project(&self) -> CSG<S> {
        flatten_and_union(self)
    }
    
    /// Slice this CSG by a plane, keeping only cross-sections on that plane.
    /// If `plane` is None, defaults to the plane z=0.
    pub fn cut(&self, plane: Option<Plane>) -> CSG<S> {

        let plane = plane.unwrap_or_else(|| Plane {
            normal: nalgebra::Vector3::new(0.0, 0.0, 1.0),
            w: 0.0,
        });
        let eps = EPSILON; // or 1e-5, as defined
    
        let mut result_polygons = Vec::new();
    
        // For each polygon in the original CSG:
        for poly in &self.polygons {
            let vcount = poly.vertices.len();
            if vcount < 2 {
                continue; // skip degenerate
            }
    
            // Classify each vertex against plane
            // side[i] = +1 if above plane, -1 if below plane, 0 if on plane
            let mut sides = Vec::with_capacity(vcount);
            for v in &poly.vertices {
                let dist = plane.normal.dot(&v.pos.coords) - plane.w;
                if dist.abs() < eps {
                    sides.push(0); // on plane
                } else if dist > 0.0 {
                    sides.push(1); // above
                } else {
                    sides.push(-1); // below
                }
            }
    
            // Collect the points where the polygon intersects the plane
            let mut intersect_points = Vec::new();
    
            for i in 0..vcount {
                let j = (i + 1) % vcount;
                let side_i = sides[i];
                let side_j = sides[j];
                let vi = &poly.vertices[i];
                let vj = &poly.vertices[j];
    
                // If a vertex lies exactly on the plane, include it
                if side_i == 0 {
                    intersect_points.push(vi.pos);
                }
    
                // If edges cross the plane, find intersection
                if side_i != side_j && side_i != 0 && side_j != 0 {
                    let denom = plane.normal.dot(&(vj.pos - vi.pos));
                    if denom.abs() > eps {
                        let t = (plane.w - plane.normal.dot(&vi.pos.coords)) / denom;
                        let new_v = vi.interpolate(vj, t).pos;
                        intersect_points.push(new_v);
                    }
                }
            }
    
            // If fewer than 3 intersection points, no valid cross-section
            if intersect_points.len() < 3 {
                continue;
            }
    
            // Sort intersection points around their average center so they form a proper polygon
            let mut avg = nalgebra::Vector3::zeros();
            for p in &intersect_points {
                avg += p.coords;
            }
            avg /= intersect_points.len() as f64;
    
            intersect_points.sort_by(|a, b| {
                let ax = a.x - avg.x;
                let ay = a.y - avg.y;
                let bx = b.x - avg.x;
                let by = b.y - avg.y;
                let angle_a = ay.atan2(ax);
                let angle_b = by.atan2(bx);
                angle_a
                    .partial_cmp(&angle_b)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
    
            // Build a new Polygon (in z=0, normal = +Z)
            let normal_2d = nalgebra::Vector3::new(0.0, 0.0, 1.0);
            let new_verts: Vec<Vertex> = intersect_points
                .into_iter()
                .map(|p| Vertex::new(nalgebra::Point3::new(p.x, p.y, 0.0), normal_2d))
                .collect();
    
            result_polygons.push(Polygon::new(new_verts, poly.shared.clone()));
        }
    
        CSG::from_polygons(result_polygons)
    }

    /// Convert a `MeshText` (from meshtext) into a list of `Polygon` in the XY plane.
    /// - `scale` allows you to resize the glyph (e.g. matching a desired font size).
    /// - By default, the glyph’s normal is set to +Z.
    fn meshtext_to_polygons(glyph_mesh: &meshtext::MeshText, scale: f64) -> Vec<Polygon<S>> {
        let mut polygons = Vec::new();
        let verts = &glyph_mesh.vertices;

        // Each set of 9 floats = one triangle: (x1,y1,z1, x2,y2,z2, x3,y3,z3)
        for tri_chunk in verts.chunks_exact(9) {
            let x1 = tri_chunk[0] as f64;
            let y1 = tri_chunk[1] as f64;
            let z1 = tri_chunk[2] as f64;
            let x2 = tri_chunk[3] as f64;
            let y2 = tri_chunk[4] as f64;
            let z2 = tri_chunk[5] as f64;
            let x3 = tri_chunk[6] as f64;
            let y3 = tri_chunk[7] as f64;
            let z3 = tri_chunk[8] as f64;

            // Scale them
            let px1 = x1 * scale;
            let py1 = y1 * scale;
            let pz1 = z1 * scale;
    
            let px2 = x2 * scale;
            let py2 = y2 * scale;
            let pz2 = z2 * scale;
    
            let px3 = x3 * scale;
            let py3 = y3 * scale;
            let pz3 = z3 * scale;
    
            // Normal = +Z
            let normal = nalgebra::Vector3::new(0.0, 0.0, 1.0);
    
            polygons.push(Polygon::new(
                vec![
                    Vertex::new(Point3::new(px1, py1, pz1), normal),
                    Vertex::new(Point3::new(px2, py2, pz2), normal),
                    Vertex::new(Point3::new(px3, py3, pz3), normal),
                ],
                None,
            ));
        }

        polygons
    }

    /// Creates 2D text in the XY plane using the `meshtext` crate to generate glyph meshes.
    ///
    /// - `text_str`: the text to render
    /// - `font_data`: TTF font file bytes (e.g. `include_bytes!("../assets/FiraMono-Regular.ttf")`)
    /// - `size`: optional scaling factor (e.g., a rough "font size"). 
    ///
    /// **Note**: This is a basic example that:
    ///   - does not handle kerning or multi-line text,
    ///   - simply advances the cursor by each glyph’s width,
    ///   - places all characters along the X axis.
    pub fn text_mesh(text_str: &str, font_data: &[u8], size: Option<f64>) -> CSG<S> {
        let mut generator = MeshGenerator::new(font_data.to_vec());
        let scale = size.unwrap_or(20.0);

        let mut all_polygons = Vec::new();
        let mut cursor_x = 0.0f64;

        for ch in text_str.chars() {
            // Optionally skip control chars
            if ch.is_control() {
                continue;
            }
            // Generate glyph mesh
            let glyph_mesh: MeshText = match generator.generate_glyph(ch, true, None) {
                Ok(m) => m,
                Err(_) => {
                    // Missing glyph? Advance by some default
                    cursor_x += scale;
                    continue;
                }
            };
    
            // Convert to polygons
            let glyph_polygons = Self::meshtext_to_polygons(&glyph_mesh, scale);

            // Translate polygons by (cursor_x, 0.0)
            let glyph_csg = CSG::from_polygons(glyph_polygons)
                .translate(Vector3::new(cursor_x, 0.0, 0.0));
            // Accumulate
            all_polygons.extend(glyph_csg.polygons);

            // Advance cursor by the glyph’s bounding-box width
            let glyph_width = glyph_mesh.bbox.max.x - glyph_mesh.bbox.min.x;
            cursor_x += glyph_width as f64 * scale;
        }

        CSG::from_polygons(all_polygons)
    }
    
    /// Re‐triangulate each polygon in this CSG using the `earclip` library (earcut).
    /// Returns a new CSG whose polygons are all triangles.
    pub fn retriangulate(&self) -> CSG<S> {
        let mut new_polygons = Vec::new();

        // For each polygon in this CSG:
        for poly in &self.polygons {
            // Skip if fewer than 3 vertices or degenerate
            if poly.vertices.len() < 3 {
                continue;
            }
            // Optional: You may also want to check if the polygon is "nearly degenerate"
            // by measuring area or normal magnitude, etc. For brevity, we skip that here.

            // 1) Build an orthonormal basis from the polygon's plane normal
            let n = poly.plane.normal.normalize();
            let (u, v) = build_orthonormal_basis(n);

            // 2) Pick a reference point on the polygon. Typically the first vertex
            let p0 = poly.vertices[0].pos;

            // 3) Project each 3D vertex into 2D coordinates: (x_i, y_i)
            // We'll store them in a flat `Vec<f64>` of the form [x0, y0, x1, y1, x2, y2, ...]
            let mut coords_2d = Vec::with_capacity(poly.vertices.len() * 2);
            for vert in &poly.vertices {
                let offset = vert.pos.coords - p0.coords;  // vector from p0 to the vertex
                let x = offset.dot(&u);
                let y = offset.dot(&v);
                coords_2d.push(x);
                coords_2d.push(y);
            }

            // 4) Call Earcut on that 2D outline. We assume no holes, so hole_indices = &[].
            //    earcut's signature is `earcut::<f64, usize>(data, hole_indices, dim)`
            //    with `dim = 2` for our XY data.
            let indices: Vec<usize> = earcut(&coords_2d, &[], 2);

            // 5) Map each returned triangle's (i0, i1, i2) back to 3D,
            //    constructing a new `Polygon` (with 3 vertices) for each tri.
            //
            //    The earcut indices are typed as `usize` in this snippet; if
            //    your crate’s `Index` is `u32`, simply adjust or cast as needed.
            for tri in indices.chunks_exact(3) {
                let mut tri_vertices = Vec::with_capacity(3);
                for &i in tri {
                    let x = coords_2d[2 * i];
                    let y = coords_2d[2 * i + 1];
                    // Inverse projection:
                    // Q_3D = p0 + x * u + y * v
                    let pos_vec = p0.coords + x * u + y * v;
                    let pos_3d  = Point3::from(pos_vec);
                    // We can store the normal = polygon's plane normal (or recalc).
                    // We'll recalc below, so for now just keep n or 0 as a placeholder:
                    tri_vertices.push(Vertex::new(pos_3d, n));
                }

                // Create a polygon from these 3 vertices. We preserve the `shared` data:
                let mut new_poly = Polygon::new(tri_vertices, poly.shared.clone());

                // Recompute the plane/normal to ensure correct orientation/shading:
                new_poly.recalc_plane_and_normals();

                new_polygons.push(new_poly);
            }
        }

        // Combine all newly formed triangles into a new CSG:
        CSG::from_polygons(new_polygons)
    }

    /// Convert the polygons in this CSG to a Parry TriMesh.
    /// Useful for collision detection or physics simulations.
    pub fn to_trimesh(&self) -> SharedShape {
        // 1) Gather all the triangles from each polygon
        // 2) Build a TriMesh from points + triangle indices
        // 3) Wrap that in a SharedShape to be used in Rapier
        let mut vertices = Vec::new();
        let mut indices = Vec::new();
        let mut index_offset = 0;

        for poly in &self.polygons {
            let tris = poly.triangulate();
            for tri in &tris {
                // Each tri is [Vertex; 3]
                //  push the positions into `vertices`
                //  build the index triplet for `indices`
                for v in tri {
                    vertices.push(Point3::new(v.pos.x, v.pos.y, v.pos.z));
                }
                indices.push([index_offset, index_offset+1, index_offset+2]);
                index_offset += 3;
            }
        }

        // TriMesh::new(Vec<[f64; 3]>, Vec<[u32; 3]>)
        let trimesh = TriMesh::new(vertices, indices).unwrap();
        SharedShape::new(trimesh)
    }

    /// Approximate mass properties using Rapier.
    pub fn mass_properties(&self, density: f64) -> (f64, Point3<f64>, Unit<Quaternion<f64>>) {
        let shape = self.to_trimesh();
        if let Some(trimesh) = shape.as_trimesh() {
            let mp = trimesh.mass_properties(density);
            (
                mp.mass(),
                mp.local_com,                         // a Point3<f64>
                mp.principal_inertia_local_frame      // a Unit<Quaternion<f64>>
            )
        } else {
	    // fallback if not a TriMesh
            (0.0, Point3::origin(), Unit::<Quaternion<f64>>::identity())
        }
    }

    /// Create a Rapier rigid body + collider from this CSG, using
    /// an axis-angle `rotation` in 3D (the vector’s length is the
    /// rotation in radians, and its direction is the axis).
    pub fn to_rigid_body(
        &self,
        rb_set: &mut RigidBodySet,
        co_set: &mut ColliderSet,
        translation: Vector3<f64>,
        rotation: Vector3<f64>, // rotation axis scaled by angle (radians)
        density: f64,
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
        let coll = ColliderBuilder::new(shape)
            .density(density)
            .build();
        co_set.insert_with_parent(coll, rb_handle, rb_set);
    
        rb_handle
    }

    // ----------------------------------------------------------
    //   Export to ASCII STL
    // ----------------------------------------------------------

    /// Convert this CSG to an **ASCII STL** string with the given `name`.
    ///
    /// ```
    /// let csg = CSG::cube(None);
    /// let stl_text = csg.to_stl("my_solid");
    /// println!("{}", stl_text);
    /// ```
    pub fn to_stl_ascii(&self, name: &str) -> String {
        let mut out = String::new();
        out.push_str(&format!("solid {}\n", name));

        for poly in &self.polygons {
            // Use the polygon plane's normal for the facet normal (normalized).
            let normal = poly.plane.normal.normalize();
            let triangles = poly.triangulate();

            for tri in triangles {
                out.push_str(&format!(
                    "  facet normal {:.6} {:.6} {:.6}\n",
                    normal.x, normal.y, normal.z
                ));
                out.push_str("    outer loop\n");
                for vertex in &tri {
                    out.push_str(&format!(
                        "      vertex {:.6} {:.6} {:.6}\n",
                        vertex.pos.x, vertex.pos.y, vertex.pos.z
                    ));
                }
                out.push_str("    endloop\n");
                out.push_str("  endfacet\n");
            }
        }

        out.push_str(&format!("endsolid {}\n", name));
        out
    }

    // ----------------------------------------------------------
    //   Export to BINARY STL (returns Vec<u8>)
    // ----------------------------------------------------------

    /// Convert this CSG to a **binary STL** byte vector with the given `name`.
    ///
    /// The resulting `Vec<u8>` can then be written to a file or handled in memory:
    ///
    /// ```
    /// let bytes = csg.to_stl_binary("my_solid")?;
    /// std::fs::write("my_solid.stl", bytes)?;
    /// ```
    pub fn to_stl_binary(&self, _name: &str) -> std::io::Result<Vec<u8>> {
        // `_name` could be embedded in the binary header if desired, but `stl_io` 
        // doesn't strictly require it. We skip storing it or store it in the 80-byte header.

        // Gather the triangles for stl_io
        let mut triangles = Vec::new();
        for poly in &self.polygons {
            let normal = poly.plane.normal.normalize();
            let tri_list = poly.triangulate();

            for tri in tri_list {
                triangles.push(stl_io::Triangle {
                    normal: stl_io::Normal::new([normal.x as f32, normal.y as f32, normal.z as f32]),
                    vertices: [
                        stl_io::Vertex::new([tri[0].pos.x as f32, tri[0].pos.y as f32, tri[0].pos.z as f32, ]),
                        stl_io::Vertex::new([tri[1].pos.x as f32, tri[1].pos.y as f32, tri[1].pos.z as f32, ]),
                        stl_io::Vertex::new([tri[2].pos.x as f32, tri[2].pos.y as f32, tri[2].pos.z as f32, ]),
                    ],
                });
            }
        }

        // Write to an in-memory buffer
        let mut cursor = Cursor::new(Vec::new());
        stl_io::write_stl(&mut cursor, triangles.iter())?;

        Ok(cursor.into_inner())
    }

    /// Import a CSG object from a binary STL file using `stl_io`.
    pub fn from_stl_file(file_path: &str) -> Result<CSG<S>, std::io::Error> {
        let mut file = OpenOptions::new().read(true).open(file_path)?;
        let stl_reader = stl_io::create_stl_reader(&mut file)?;

        let mut polygons = Vec::new();

        for tri_result in stl_reader {
            // Handle potential errors from the STL reader
            let tri = match tri_result {
                Ok(t) => t,
                Err(e) => return Err(e), // Propagate the error
            };

            // Construct vertices and a polygon
            let vertices = vec![
                Vertex::new(
                    Point3::new(tri.vertices[0][0] as f64, tri.vertices[0][1] as f64, tri.vertices[0][2] as f64),
                    Vector3::new(tri.normal[0] as f64, tri.normal[1] as f64, tri.normal[2] as f64),
                ),
                Vertex::new(
                    Point3::new(tri.vertices[1][0] as f64, tri.vertices[1][1] as f64, tri.vertices[1][2] as f64),
                    Vector3::new(tri.normal[0] as f64, tri.normal[1] as f64, tri.normal[2] as f64),
                ),
                Vertex::new(
                    Point3::new(tri.vertices[2][0] as f64, tri.vertices[2][1] as f64, tri.vertices[2][2] as f64),
                    Vector3::new(tri.normal[0] as f64, tri.normal[1] as f64, tri.normal[2] as f64),
                ),
            ];
            polygons.push(Polygon::new(vertices, None));
        }

        Ok(CSG::from_polygons(polygons))
    }
}

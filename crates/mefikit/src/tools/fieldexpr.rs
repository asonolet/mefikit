//! Field expression system for computing derived fields.
//!
//! Provides a domain-specific language for building and evaluating
//! field expressions using mathematical operations.

use ndarray::{self as nd};
use smallvec::SmallVec;
use std::{
    collections::BTreeSet,
    ops::{Add, Div, Mul, Sub},
    sync::Arc,
};

use super::centroids::{centroids, x_center, y_center, z_center};
use super::measure::measure;
use super::normals::{normals, nx as normal_x, ny as normal_y, nz as normal_z};
use crate::mesh::{Dimension, FieldArcD, FieldCowD, FieldOwnedD, UMesh, UMeshBase, UMeshView};

/// An expression tree for field computations.
#[derive(Clone, Debug)]
pub enum FieldExpr {
    /// A broadcastable constant array.
    Array(nd::Array<f64, nd::IxDyn>),
    /// A reference to a named field in the mesh.
    Field(String),
    /// A binary operation between two expressions.
    BinaryExpr {
        operator: BinaryOp,
        left: Arc<FieldExpr>,
        right: Arc<FieldExpr>,
    },
    /// A unary operation on an expression.
    UnaryExpr {
        operator: UnaryOp,
        expr: Arc<FieldExpr>,
    },
    /// Element measure (not yet implemented).
    Measure,
    /// Element centroids (not yet implemented).
    Centroid,
    /// X coordinate (not yet implemented).
    X,
    /// Y coordinate (not yet implemented).
    Y,
    /// Z coordinate (not yet implemented).
    Z,
    /// Index into a multi-component field.
    Index(Arc<FieldExpr>, SmallVec<[usize; 2]>),
    /// Surface normal (3-vector for 2D cells in 3D, in-plane 2-vector for 1D cells in 2D).
    Normal,
    /// X component of the surface normal.
    Nx,
    /// Y component of the surface normal.
    Ny,
    /// Z component of the surface normal.
    Nz,
}

/// Binary operations available in field expressions.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum BinaryOp {
    /// Addition.
    Add,
    /// Multiplication.
    Mul,
    /// Subtraction.
    Sub,
    /// Division.
    Div,
    /// Power (a^b).
    Pow,
    /// Matrix product (shorthand `@`).
    MatMul,
}

/// Unary operations available in field expressions.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum UnaryOp {
    /// Sine function.
    Sin,
    /// Square root.
    Sqrt,
    /// Squaring (x^2).
    Square,
    /// Cosine function.
    Cos,
    /// Exponential function.
    Exp,
    /// Natural logarithm.
    Ln,
    /// Base-10 logarithm.
    Log10,
    /// Absolute value.
    Abs,
    /// Tangent function.
    Tan,
}

impl FieldExpr {
    /// Applies the sine function to this expression.
    pub fn sin(self) -> Self {
        Self::UnaryExpr {
            operator: UnaryOp::Sin,
            expr: Arc::new(self),
        }
    }

    /// Applies the cosine function to this expression.
    pub fn cos(self) -> Self {
        Self::UnaryExpr {
            operator: UnaryOp::Cos,
            expr: Arc::new(self),
        }
    }

    /// Applies the square root to this expression.
    pub fn sqrt(self) -> Self {
        Self::UnaryExpr {
            operator: UnaryOp::Sqrt,
            expr: Arc::new(self),
        }
    }

    /// Squares this expression.
    pub fn square(self) -> Self {
        Self::UnaryExpr {
            operator: UnaryOp::Square,
            expr: Arc::new(self),
        }
    }

    /// Applies the exponential function to this expression.
    pub fn exp(self) -> Self {
        Self::UnaryExpr {
            operator: UnaryOp::Exp,
            expr: Arc::new(self),
        }
    }

    /// Applies the natural logarithm to this expression.
    pub fn ln(self) -> Self {
        Self::UnaryExpr {
            operator: UnaryOp::Ln,
            expr: Arc::new(self),
        }
    }

    /// Applies the base-10 logarithm to this expression.
    pub fn log10(self) -> Self {
        Self::UnaryExpr {
            operator: UnaryOp::Log10,
            expr: Arc::new(self),
        }
    }

    /// Applies the tangent function to this expression.
    pub fn tan(self) -> Self {
        Self::UnaryExpr {
            operator: UnaryOp::Tan,
            expr: Arc::new(self),
        }
    }

    /// Applies the absolute value to this expression.
    pub fn abs(self) -> Self {
        Self::UnaryExpr {
            operator: UnaryOp::Abs,
            expr: Arc::new(self),
        }
    }

    /// Raises this expression to the power of `other`.
    pub fn pow(self, other: Self) -> Self {
        Self::BinaryExpr {
            operator: BinaryOp::Pow,
            left: Arc::new(self),
            right: Arc::new(other),
        }
    }

    /// Computes the matrix product (shorthand `@`) of this expression and `other`.
    pub fn matmul(self, other: Self) -> Self {
        Self::BinaryExpr {
            operator: BinaryOp::MatMul,
            left: Arc::new(self),
            right: Arc::new(other),
        }
    }
}

/// Creates a field expression referencing a named field.
pub fn field(name: &str) -> FieldExpr {
    FieldExpr::Field(name.to_owned())
}

/// Creates a field expression from a constant array.
pub fn arr<D: nd::Dimension>(arr: nd::Array<f64, D>) -> FieldExpr {
    FieldExpr::Array(arr.into_dyn())
}

/// Creates a field expression for the surface normal (3-vector for 2D cells in 3D,
/// in-plane 2-vector for 1D cells in 2D).
pub fn normal() -> FieldExpr {
    FieldExpr::Normal
}

/// Creates a field expression for the X component of the surface normal.
pub fn nx() -> FieldExpr {
    FieldExpr::Nx
}

/// Creates a field expression for the Y component of the surface normal.
pub fn ny() -> FieldExpr {
    FieldExpr::Ny
}

/// Creates a field expression for the Z component of the surface normal.
pub fn nz() -> FieldExpr {
    FieldExpr::Nz
}

impl Add for FieldExpr {
    type Output = FieldExpr;

    fn add(self, rhs: FieldExpr) -> FieldExpr {
        FieldExpr::BinaryExpr {
            operator: BinaryOp::Add,
            left: Arc::new(self),
            right: Arc::new(rhs),
        }
    }
}

impl Sub for FieldExpr {
    type Output = FieldExpr;

    fn sub(self, rhs: FieldExpr) -> FieldExpr {
        FieldExpr::BinaryExpr {
            operator: BinaryOp::Sub,
            left: Arc::new(self),
            right: Arc::new(rhs),
        }
    }
}

impl Mul for FieldExpr {
    type Output = FieldExpr;

    fn mul(self, rhs: FieldExpr) -> FieldExpr {
        FieldExpr::BinaryExpr {
            operator: BinaryOp::Mul,
            left: Arc::new(self),
            right: Arc::new(rhs),
        }
    }
}

impl Div for FieldExpr {
    type Output = FieldExpr;

    fn div(self, rhs: FieldExpr) -> FieldExpr {
        FieldExpr::BinaryExpr {
            operator: BinaryOp::Div,
            left: Arc::new(self),
            right: Arc::new(rhs),
        }
    }
}

impl FieldExpr {
    /// Selects a component from a multi-component field.
    pub fn index(self, index: &[usize]) -> Self {
        Self::Index(Arc::new(self), index.into())
    }
}

/// Trait for evaluating field expressions on a mesh.
pub trait Evaluable {
    /// Evaluates the expression on the given mesh and returns the result as a field.
    fn evaluate<'a>(&'a self, mesh: &'a UMeshView<'a>, dim: Option<Dimension>) -> FieldCowD<'a>;
}

/// Handles for a referenced field name, all element types (and thus dimensions) that
/// actually store it.
fn field_storage_dims(mesh: &UMeshView, name: &str) -> BTreeSet<Dimension> {
    mesh.blocks()
        .filter(|(_, b)| b.fields.contains_key(name))
        .map(|(et, _)| et.dimension())
        .collect()
}

/// Collects the dimension hints carried by an expression tree.
fn collect_dim_hints(
    expr: &FieldExpr,
    mesh: &UMeshView,
    found_normal: &mut bool,
    field_dims: &mut BTreeSet<Dimension>,
) {
    match expr {
        FieldExpr::Normal | FieldExpr::Nx | FieldExpr::Ny | FieldExpr::Nz => {
            *found_normal = true;
        }
        FieldExpr::Field(name) => {
            field_dims.extend(field_storage_dims(mesh, name));
        }
        FieldExpr::BinaryExpr { left, right, .. } => {
            collect_dim_hints(left, mesh, found_normal, field_dims);
            collect_dim_hints(right, mesh, found_normal, field_dims);
        }
        FieldExpr::UnaryExpr { expr, .. } | FieldExpr::Index(expr, _) => {
            collect_dim_hints(expr, mesh, found_normal, field_dims);
        }
        FieldExpr::Array(_)
        | FieldExpr::Measure
        | FieldExpr::Centroid
        | FieldExpr::X
        | FieldExpr::Y
        | FieldExpr::Z => {}
    }
}

/// Infers the target evaluation dimension for a field expression evaluated with
/// `dim = None`.
///
/// The expression advertises the dimension it should be evaluated on:
/// - `Normal`/`Nx`/`Ny`/`Nz` are defined on the hypersurface, `space_dimension() - 1`;
/// - each referenced field is defined on the element types (and hence dimensions) that
///   actually store it;
/// - with no hints, the highest element dimension (the topological dimension) is used.
///
/// If the hints disagree there is no single consistent target, so this panics rather than
/// silently evaluating parts of the expression on different dimensions.
fn infer_dim(mesh: &UMeshView, expr: &FieldExpr) -> Dimension {
    let mut found_normal = false;
    let mut field_dims: BTreeSet<Dimension> = BTreeSet::new();
    collect_dim_hints(expr, mesh, &mut found_normal, &mut field_dims);

    let mut candidates: BTreeSet<Dimension> = BTreeSet::new();
    if found_normal {
        let hyper = Dimension::try_from(mesh.space_dimension() - 1)
            .expect("space dimension minus one is a valid element dimension");
        candidates.insert(hyper);
    }
    candidates.extend(field_dims.iter().copied());

    if candidates.len() == 1 {
        return *candidates.iter().next().unwrap();
    }
    if candidates.is_empty() {
        return mesh
            .topological_dimension()
            .expect("cannot infer a target dimension: the mesh has no elements");
    }
    panic!(
        "cannot infer a single target dimension for this field expression: found mixed \
         dimensions {candidates:?} (normals live on the hypersurface and referenced fields \
         on their storage dimension); pass an explicit dimension"
    );
}

impl Evaluable for FieldExpr {
    fn evaluate<'a>(&'a self, mesh: &'a UMeshView<'a>, dim: Option<Dimension>) -> FieldCowD<'a> {
        let dim = match dim {
            Some(d) => d,
            None => infer_dim(mesh, self),
        };
        let elems: Vec<_> = mesh
            .element_types()
            .filter(|et| et.dimension() == dim)
            .cloned()
            .collect();
        match self {
            FieldExpr::Array(arr) => FieldCowD::from_array(arr.view().into(), elems.as_slice()),
            FieldExpr::Field(name) => mesh.field(name, Some(dim)).unwrap().into(),
            FieldExpr::BinaryExpr {
                operator,
                left,
                right,
            } => {
                let left_eval = left.evaluate(mesh, Some(dim));
                let right_eval = right.evaluate(mesh, Some(dim));
                match operator {
                    BinaryOp::Add => (&left_eval + &right_eval).into(),
                    BinaryOp::Sub => (&left_eval - &right_eval).into(),
                    BinaryOp::Mul => (&left_eval * &right_eval).into(),
                    BinaryOp::Div => (&left_eval / &right_eval).into(),
                    BinaryOp::Pow => left_eval
                        .map_zip_broadcast(&right_eval, |a, b| a.powf(b))
                        .into(),
                    BinaryOp::MatMul => left_eval.map_matmul(&right_eval).into(),
                }
            }
            FieldExpr::UnaryExpr { operator, expr } => {
                let expr_eval = expr.evaluate(mesh, Some(dim));
                match operator {
                    UnaryOp::Sin => expr_eval.mapv(|x| x.sin()).into(),
                    UnaryOp::Cos => expr_eval.mapv(|x| x.cos()).into(),
                    UnaryOp::Tan => expr_eval.mapv(|x| x.tan()).into(),
                    UnaryOp::Sqrt => expr_eval.mapv(|x| x.sqrt()).into(),
                    UnaryOp::Square => expr_eval.mapv(|x| x.powi(2)).into(),
                    UnaryOp::Exp => expr_eval.mapv(|x| x.exp()).into(),
                    UnaryOp::Ln => expr_eval.mapv(|x| x.ln()).into(),
                    UnaryOp::Log10 => expr_eval.mapv(|x| x.log10()).into(),
                    UnaryOp::Abs => expr_eval.mapv(|x| x.abs()).into(),
                }
            }
            FieldExpr::Measure => FieldOwnedD::new(
                measure(mesh, Some(dim))
                    .into_iter()
                    .map(|(k, v)| (k, v.into_dyn()))
                    .collect(),
            )
            .into(),
            FieldExpr::Centroid => FieldOwnedD::new(
                centroids(mesh, Some(dim))
                    .into_iter()
                    .map(|(k, v)| (k, v.into_dyn()))
                    .collect(),
            )
            .into(),
            FieldExpr::X => FieldOwnedD::new(
                x_center(mesh, Some(dim))
                    .into_iter()
                    .map(|(k, v)| (k, v.into_dyn()))
                    .collect(),
            )
            .into(),
            FieldExpr::Y => FieldOwnedD::new(
                y_center(mesh, Some(dim))
                    .into_iter()
                    .map(|(k, v)| (k, v.into_dyn()))
                    .collect(),
            )
            .into(),
            FieldExpr::Z => FieldOwnedD::new(
                z_center(mesh, Some(dim))
                    .into_iter()
                    .map(|(k, v)| (k, v.into_dyn()))
                    .collect(),
            )
            .into(),
            // FieldExpr::Rcyl => mesh.coords().slice(nd::s![.., 0]).to_owned(),
            // FieldExpr::Rsph => mesh.coords().slice(nd::s![.., 0]).to_owned(),
            // FieldExpr::Theta => mesh.coords().slice(nd::s![.., 1]).to_owned(),
            // FieldExpr::Phi => mesh.coords().slice(nd::s![.., 2]).to_owned(),
            FieldExpr::Index(expr, index) => {
                let eval = expr.evaluate(mesh, Some(dim));
                let idx = index[0];
                FieldOwnedD::new(
                    eval.0
                        .iter()
                        .map(|(k, v)| {
                            let last = v.ndim() - 1;
                            (*k, v.index_axis(nd::Axis(last), idx).to_owned())
                        })
                        .collect(),
                )
                .into()
            }
            FieldExpr::Normal => FieldOwnedD::new(
                normals(mesh, Some(dim))
                    .into_iter()
                    .map(|(k, v)| (k, v.into_dyn()))
                    .collect(),
            )
            .into(),
            FieldExpr::Nx => FieldOwnedD::new(
                normal_x(mesh, Some(dim))
                    .into_iter()
                    .map(|(k, v)| (k, v.into_dyn()))
                    .collect(),
            )
            .into(),
            FieldExpr::Ny => FieldOwnedD::new(
                normal_y(mesh, Some(dim))
                    .into_iter()
                    .map(|(k, v)| (k, v.into_dyn()))
                    .collect(),
            )
            .into(),
            FieldExpr::Nz => FieldOwnedD::new(
                normal_z(mesh, Some(dim))
                    .into_iter()
                    .map(|(k, v)| (k, v.into_dyn()))
                    .collect(),
            )
            .into(),
        }
    }
}

/// Trait for evaluating field expressions on a mesh.
pub trait MeshEvaluable {
    /// Evaluates an expression and returns the result as a new field.
    fn eval_field(&self, dim: Option<Dimension>, expr: FieldExpr) -> FieldOwnedD;
}

/// Trait for evaluating and storing field expressions.
pub trait MeshEvalUpdatable: MeshEvaluable {
    /// Evaluates an expression and stores the result as a named field in the mesh.
    fn eval_update_field(
        &mut self,
        name: &str,
        dim: Option<Dimension>,
        expr: FieldExpr,
    ) -> Option<FieldArcD>;
}

impl<N, C, F, G> MeshEvaluable for UMeshBase<N, C, F, G>
where
    N: nd::Data<Elem = f64>,
    C: nd::Data<Elem = usize>,
    F: nd::Data<Elem = f64>,
    G: nd::Data<Elem = usize>,
{
    fn eval_field(&self, dim: Option<Dimension>, expr: FieldExpr) -> FieldOwnedD {
        expr.evaluate(&self.view(), dim).to_owned()
    }
}

impl MeshEvalUpdatable for UMesh {
    fn eval_update_field(
        &mut self,
        name: &str,
        dim: Option<Dimension>,
        expr: FieldExpr,
    ) -> Option<FieldArcD> {
        let field = self.eval_field(dim, expr);
        self.update_field(name, field.into_shared())
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::mesh::{ElementType, FieldArcD};
    use crate::mesh_examples as me;
    use crate::prelude as mf;
    use crate::tools::Measurable;
    use approx::*;
    use ndarray as nd;
    use std::collections::BTreeMap;

    fn hex8_quad4_mesh() -> mf::UMesh {
        let coords = nd::Array2::from_shape_vec(
            (9, 3),
            vec![
                0.0, 0.0, 0.0, // 0
                1.0, 0.0, 0.0, // 1
                1.0, 1.0, 0.0, // 2
                0.0, 1.0, 0.0, // 3
                0.0, 0.0, 1.0, // 4
                1.0, 0.0, 1.0, // 5
                1.0, 1.0, 1.0, // 6
                0.0, 1.0, 1.0, // 7
                0.0, 0.0, 2.0, // 8
            ],
        )
        .unwrap();
        let mut mesh = mf::UMesh::new(coords.into());
        mesh.add_regular_block(
            mf::ElementType::HEX8,
            nd::arr2(&[[0, 1, 2, 3, 4, 5, 6, 7]]).to_shared(),
            None,
        );
        mesh.add_regular_block(
            mf::ElementType::QUAD4,
            nd::arr2(&[[4, 5, 6, 7]]).to_shared(),
            None,
        );
        mesh
    }

    #[test]
    fn compose_expr() {
        let a = field("toto");
        let b = field("exponent");
        let c = arr(nd::arr0(1.0));
        let _res = a.pow(b) + c;
    }

    #[test]
    fn measure_squared() {
        let mut m = me::make_imesh_2d(10);
        m.measure_update("M", None);
        let mes_squared5 = field("M").square() * arr(nd::arr0(5.0));
        m.eval_field(None, mes_squared5);
    }

    #[test]
    fn test_field_expr_sin() {
        let expr = arr(nd::arr0(0.0)).sin();
        match expr {
            FieldExpr::UnaryExpr { operator, .. } => assert_eq!(operator, UnaryOp::Sin),
            _ => panic!("Expected UnaryExpr"),
        }
    }

    #[test]
    fn test_field_expr_cos() {
        let expr = arr(nd::arr0(0.0)).cos();
        match expr {
            FieldExpr::UnaryExpr { operator, .. } => assert_eq!(operator, UnaryOp::Cos),
            _ => panic!("Expected UnaryExpr"),
        }
    }

    #[test]
    fn test_field_expr_sqrt() {
        let expr = arr(nd::arr0(4.0)).sqrt();
        match expr {
            FieldExpr::UnaryExpr { operator, .. } => assert_eq!(operator, UnaryOp::Sqrt),
            _ => panic!("Expected UnaryExpr"),
        }
    }

    #[test]
    fn test_field_expr_square() {
        let expr = arr(nd::arr0(3.0)).square();
        match expr {
            FieldExpr::UnaryExpr { operator, .. } => assert_eq!(operator, UnaryOp::Square),
            _ => panic!("Expected UnaryExpr"),
        }
    }

    #[test]
    fn test_field_expr_exp() {
        let expr = arr(nd::arr0(1.0)).exp();
        match expr {
            FieldExpr::UnaryExpr { operator, .. } => assert_eq!(operator, UnaryOp::Exp),
            _ => panic!("Expected UnaryExpr"),
        }
    }

    #[test]
    fn test_field_expr_ln() {
        let expr = arr(nd::arr0(1.0)).ln();
        match expr {
            FieldExpr::UnaryExpr { operator, .. } => assert_eq!(operator, UnaryOp::Ln),
            _ => panic!("Expected UnaryExpr"),
        }
    }

    #[test]
    fn test_field_expr_log10() {
        let expr = arr(nd::arr0(1.0)).log10();
        match expr {
            FieldExpr::UnaryExpr { operator, .. } => assert_eq!(operator, UnaryOp::Log10),
            _ => panic!("Expected UnaryExpr"),
        }
    }

    #[test]
    fn test_field_expr_tan() {
        let expr = arr(nd::arr0(0.0)).tan();
        match expr {
            FieldExpr::UnaryExpr { operator, .. } => assert_eq!(operator, UnaryOp::Tan),
            _ => panic!("Expected UnaryExpr"),
        }
    }

    #[test]
    fn test_field_expr_abs() {
        let expr = arr(nd::arr0(-1.0)).abs();
        match expr {
            FieldExpr::UnaryExpr { operator, .. } => assert_eq!(operator, UnaryOp::Abs),
            _ => panic!("Expected UnaryExpr"),
        }
    }

    #[test]
    fn test_binary_expr_add() {
        let a = field("A");
        let b = field("B");
        let expr = a + b;
        match expr {
            FieldExpr::BinaryExpr { operator, .. } => assert_eq!(operator, BinaryOp::Add),
            _ => panic!("Expected BinaryExpr"),
        }
    }

    #[test]
    fn test_binary_expr_mul() {
        let a = field("A");
        let b = field("B");
        let expr = a * b;
        match expr {
            FieldExpr::BinaryExpr { operator, .. } => assert_eq!(operator, BinaryOp::Mul),
            _ => panic!("Expected BinaryExpr"),
        }
    }

    #[test]
    fn test_binary_expr_sub() {
        let a = field("A");
        let b = field("B");
        let expr = a - b;
        match expr {
            FieldExpr::BinaryExpr { operator, .. } => assert_eq!(operator, BinaryOp::Sub),
            _ => panic!("Expected BinaryExpr"),
        }
    }

    #[test]
    fn test_binary_expr_div() {
        let a = field("A");
        let b = field("B");
        let expr = a / b;
        match expr {
            FieldExpr::BinaryExpr { operator, .. } => assert_eq!(operator, BinaryOp::Div),
            _ => panic!("Expected BinaryExpr"),
        }
    }

    #[test]
    fn test_binary_expr_matmul() {
        let a = field("A");
        let b = field("B");
        let expr = a.matmul(b);
        match expr {
            FieldExpr::BinaryExpr { operator, .. } => assert_eq!(operator, BinaryOp::MatMul),
            _ => panic!("Expected BinaryExpr"),
        }
    }

    #[test]
    fn test_index_expr() {
        let e = field("toto").index(&[0]);
        assert!(matches!(e, FieldExpr::Index(..)));
    }

    #[test]
    fn test_normal_exprs() {
        assert!(matches!(normal(), FieldExpr::Normal));
        assert!(matches!(nx(), FieldExpr::Nx));
        assert!(matches!(ny(), FieldExpr::Ny));
        assert!(matches!(nz(), FieldExpr::Nz));
    }

    #[test]
    fn test_eval_field() {
        let mut mesh = me::make_imesh_2d(5);
        mesh.measure_update("area", None);
        let expr = field("area").square();
        let result = mesh.eval_field(None, expr);
        assert!(result.0.contains_key(&ElementType::QUAD4));
    }

    #[test]
    fn test_eval_update_field() {
        let mut mesh = me::make_imesh_2d(5);
        mesh.measure_update("area", None);
        let expr = field("area") * arr(nd::arr0(2.0));
        let _result = mesh.eval_update_field("doubled", None, expr);
        // eval_update_field returns None when the field is new (not replaced)
        assert!(mesh.field("doubled", None).is_some());
    }

    #[test]
    fn test_eval_normal() {
        // A planar quad in 3D with unit normal (0, 0, 1).
        let coords = nd::Array2::from_shape_vec(
            (4, 3),
            vec![
                0.0, 0.0, 0.0, //
                1.0, 0.0, 0.0, //
                0.0, 1.0, 0.0, //
                1.0, 1.0, 0.0, //
            ],
        )
        .unwrap();
        let mut mesh = crate::mesh::UMesh::new(coords.into());
        mesh.add_regular_block(
            crate::mesh::ElementType::QUAD4,
            nd::arr2(&[[0, 1, 3, 2]]).to_shared(),
            None,
        );
        let result = mesh.eval_field(None, normal());
        let block = result.0.get(&ElementType::QUAD4).unwrap();
        assert_eq!(block.shape(), &[1, 3]);
        assert!((block[[0, 2]] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn infer_dim_normal_matches_boundary_field() {
        // A field stored on the QUAD4 boundary combined with Normal must be inferred to
        // the hypersurface dimension (space_dim - 1 = 2), and both evaluated on QUAD4.
        let mut mesh = hex8_quad4_mesh();
        let quad_field: BTreeMap<ElementType, nd::ArcArray<f64, nd::IxDyn>> =
            [(ElementType::QUAD4, nd::arr1(&[2.0]).to_shared().into_dyn())]
                .into_iter()
                .collect();
        mesh.update_field("quad_val", FieldArcD::new(quad_field));

        let expr = normal() * field("quad_val");
        let result = mesh.eval_field(None, expr);
        assert!(
            result.0.contains_key(&ElementType::QUAD4),
            "expected a QUAD4 result block"
        );
        assert!(
            !result.0.contains_key(&ElementType::HEX8),
            "volume cells must not appear in a normals expression"
        );
        let block = result.0.get(&ElementType::QUAD4).unwrap();
        assert_eq!(block.shape(), &[1, 3]);
        assert_abs_diff_eq!(block[[0, 0]], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(block[[0, 1]], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(block[[0, 2]], 2.0, epsilon = 1e-12);
    }

    #[test]
    #[should_panic(expected = "mixed")]
    fn infer_dim_mixed_normal_and_volume_field_panics() {
        // A volume field on HEX8 combined with Normal cannot be evaluated at a single
        // uniform dimension, so inference must error.
        let mut mesh = hex8_quad4_mesh();
        let hex_field: BTreeMap<ElementType, nd::ArcArray<f64, nd::IxDyn>> =
            [(ElementType::HEX8, nd::arr1(&[3.0]).to_shared().into_dyn())]
                .into_iter()
                .collect();
        mesh.update_field("vol_val", FieldArcD::new(hex_field));

        let expr = normal() * field("vol_val");
        let _ = mesh.eval_field(None, expr);
    }

    #[test]
    fn explicit_dim_is_strict() {
        // An explicit dimension is honored strictly, without inference.
        let mut mesh = hex8_quad4_mesh();
        let quad_field: BTreeMap<ElementType, nd::ArcArray<f64, nd::IxDyn>> =
            [(ElementType::QUAD4, nd::arr1(&[2.0]).to_shared().into_dyn())]
                .into_iter()
                .collect();
        mesh.update_field("quad_val", FieldArcD::new(quad_field));

        let expr = nz() * field("quad_val");
        let result = mesh.eval_field(Some(Dimension::D2), expr);
        assert!(result.0.contains_key(&ElementType::QUAD4));
        assert!(!result.0.contains_key(&ElementType::HEX8));
        assert_abs_diff_eq!(result.0[&ElementType::QUAD4][0], 2.0, epsilon = 1e-12);
    }
}

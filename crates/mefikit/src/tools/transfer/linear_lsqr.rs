use std::num::NonZero;

use kiddo::{ImmutableKdTree, dist::SquaredEuclidean};
use ndarray::{Array2, ArrayD, ArrayView2, ArrayViewD, Zip};

pub struct LinearInterpTransfer {
    /// Interpolation operator
    indices: Array2<usize>, // (n_tgt × k)
    weights: Array2<f64>, // (n_tgt × k)
}

impl LinearInterpTransfer {
    pub fn from_coords(
        src_coords: &ArrayView2<f64>,
        tgt_coords: &ArrayView2<f64>,
        k: usize,
    ) -> Self {
        let dim = src_coords.ncols();
        assert_eq!(dim, 3); // simplify for now

        let source = as_points3(src_coords).unwrap();

        let tree = ImmutableKdTree::new_from_slice(source).unwrap();

        let n_tgt = tgt_coords.nrows();

        let mut indices: Array2<usize> = Array2::zeros((n_tgt, k));
        let mut weights = Array2::zeros((n_tgt, k));

        for (j, p) in tgt_coords.outer_iter().enumerate() {
            //------------------------------------------
            // 1. k nearest neighbours
            //------------------------------------------

            let nn = tree
                .query(&[p[0], p[1], p[2]])
                .nearest_n::<SquaredEuclidean<f64>>(NonZero::new(k).unwrap())
                .execute();

            //------------------------------------------
            // 2. Compute interpolation weights
            //------------------------------------------

            // Placeholder
            let w = vec![1.0; k];

            for l in 0..k {
                indices[[j, l]] = nn[l].item as usize;
            }

            // TODO:
            //
            // Compute weights from
            //   self.src_coords
            //   target point p
            //   neighbour indices
            //
            // Store them in w[]

            for l in 0..k {
                weights[[j, l]] = w[l];
            }
        }
        Self { indices, weights }
    }

    pub fn apply_on_array(&self, src_field: &ArrayViewD<f64>) -> ArrayD<f64> {
        let n_src = src_field.shape()[0];
        assert_eq!(n_src, self.indices.nrows());

        let n_compo = src_field.len() / n_src;

        let n_tgt = self.indices.nrows();
        let k = self.indices.ncols();

        let mut tgt = Array2::<f64>::zeros((n_tgt, n_compo));
        let src_view = src_field
            .view()
            .into_shape_with_order((n_src, n_compo))
            .unwrap();

        for j in 0..n_tgt {
            for l in 0..k {
                let i = self.indices[[j, l]];
                let w = self.weights[[j, l]];
                // accumulate the whole row
                Zip::from(tgt.row_mut(j))
                    .and(src_view.row(i))
                    .for_each(|d, &s| {
                        *d += w * s;
                    });
            }
        }

        let mut tgt_shape = src_field.raw_dim();
        tgt_shape[0] = n_tgt;
        tgt.into_shape_with_order(tgt_shape).unwrap()
    }
}

fn as_points3<'a, 'b>(coords: &'a ArrayView2<'b, f64>) -> Option<&'b [[f64; 3]]>
where
    'b: 'a,
{
    if coords.ncols() != 3 {
        return None;
    }

    let slice = coords.as_slice()?;

    // Safety:
    // - slice length is 3*n
    // - f64 has the same alignment
    // - [[f64;3]] is a contiguous array of 3 f64 values
    let ptr = slice.as_ptr() as *const [f64; 3];

    let len = slice.len() / 3;

    Some(unsafe { std::slice::from_raw_parts(ptr, len) })
}

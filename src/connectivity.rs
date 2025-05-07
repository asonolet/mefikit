use ndarray::{s, Array1, Array2, ArrayView1, ArrayViewMut1, Axis};
// use rayon::prelude::*;

pub enum Connectivity {
    Regular(Array2<usize>),
    Poly {
        data: Array1<usize>,
        offsets: Array1<usize>,
    },
}

pub struct PolyConnIterator<'a> {
    data: &'a Array1<usize>,
    offsets: &'a Array1<usize>,
    index: usize,
}

impl<'a> Iterator for PolyConnIterator<'a> {
    type Item = ArrayView1<'a, usize>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.offsets.len() - 1 {
            return None;
        }
        let start = self.offsets[self.index];
        let end = self.offsets[self.index + 1];
        let result = self.data.slice(s![start..end]);
        self.index += 1;
        Some(result)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.offsets.len() - 1;
        (len, Some(len))
    }
}

// pub struct PolyConnIteratorMut<'a> {
//     data: ArrayViewMut1<'a, usize>,
//     offsets: &'a Array1<usize>,
//     index: usize,
// }
// 
// impl<'a> Iterator for PolyConnIteratorMut<'a> {
//     type Item = ArrayViewMut1<'a, usize>;
// 
//     fn next(&mut self) -> Option<Self::Item> {
//         let PolyConnIteratorMut { data, offsets, index } = self;
//         if *index >= offsets.len() - 1 {
//             return None;
//         }
//         let start = offsets[*index];
//         let end = offsets[*index + 1];
//         let result = data.slice_mut(s![start..end]);
//         *index += 1;
//         Some(result)
//     }
// 
//     fn size_hint(&self) -> (usize, Option<usize>) {
//         let len = self.offsets.len() - 1;
//         (len, Some(len))
//     }
// }
//

enum ConnectivityIterator<'a> {
    Regular(ndarray::iter::AxisIter<'a, usize, ndarray::Dim<[usize; 1]>>),
    Poly(PolyConnIterator<'a>),
}

impl<'a> Iterator for ConnectivityIterator<'a> {
    type Item = ArrayView1<'a, usize>;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            ConnectivityIterator::Regular(iter) => iter.next(),
            ConnectivityIterator::Poly(iter) => iter.next(),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        match self {
            ConnectivityIterator::Regular(iter) => iter.size_hint(),
            ConnectivityIterator::Poly(iter) => iter.size_hint(),
        }
    }
}

impl Connectivity {
    pub fn new_regular(conn: Array2<usize>) -> Self {
        Connectivity::Regular(conn)
    }

    pub fn new_poly(data: Array1<usize>, offsets: Array1<usize>) -> Self {
        Connectivity::Poly { data, offsets }
    }
}

impl Connectivity {
    pub fn len(&self) -> usize {
        match self {
            Connectivity::Regular(conn) => conn.nrows(),
            Connectivity::Poly { offsets, .. } => offsets.len() - 1,
        }
    }

    pub fn get(&self, index: usize) -> ArrayView1<'_, usize> {
        match self {
            Connectivity::Regular(conn) => conn.row(index),
            Connectivity::Poly { data, offsets } => {
                let start = offsets[index];
                let end = offsets[index + 1];
                data.slice(s![start..end])
            }
        }
    }
    pub fn get_mut(&mut self, index: usize) -> ArrayViewMut1<'_, usize> {
        match self {
            Connectivity::Regular(conn) => conn.row_mut(index),
            Connectivity::Poly { data, offsets } => {
                let start = offsets[index];
                let end = offsets[index + 1];
                data.slice_mut(s![start..end])
            }
        }
    }

    pub fn iter<'a>(&'a self) -> impl Iterator<Item = ArrayView1<'a, usize>> + 'a {
        match self {
            Connectivity::Regular(conn) => {
                ConnectivityIterator::Regular(conn.axis_iter(Axis(0)))
            }
            Connectivity::Poly { data, offsets } => {
                ConnectivityIterator::Poly(PolyConnIterator {
                    data,
                    offsets,
                    index: 0,
                })
            }
        }
    }

}

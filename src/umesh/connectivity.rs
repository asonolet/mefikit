use ndarray::{s, Array1, Array2, ArrayView1, ArrayViewMut1, Axis};
// use rayon::prelude::*;

/// Connectivity structure to represent the connectivity of a mesh.
///
/// It can be either regular or polygonal. Regular connectivity is represented as a 2D array,
/// while polygonal connectivity is represented as a 1D array with offsets. The offsets array
/// indicates the start and end of each polygon in the data array. The data array contains the
/// indices of the vertices of the polygons.
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

pub struct PolyConnIteratorMut<'a> {
    data: &'a mut [usize],
    offsets: &'a [usize],
    index: usize,
}

impl<'a> Iterator for PolyConnIteratorMut<'a> {
    type Item = ArrayViewMut1<'a, usize>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.offsets.len() - 1 {
            return None;
        }
        let start = self.offsets[self.index];
        let end = self.offsets[self.index + 1];
        let len = end - start;
        self.index += 1;

        // Split off the first `len` elements from `self.data`
        // using `split_at_mut`
        // Then replace the data slice with the rest of the data.
        let data = std::mem::take(&mut self.data);

        let (chunk, rest) = data.split_at_mut(len);

        // Update the data slice to point to the rest of the data
        self.data = rest;
        // Return the chunk
        // The chunk is a mutable slice of the original data
        // and the rest is the remaining data
        Some(ArrayViewMut1::from(chunk))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.offsets.len() - 1;
        (len, Some(len))
    }
}

enum ConnectivityIteratorMut<'a> {
    Regular(ndarray::iter::AxisIterMut<'a, usize, ndarray::Dim<[usize; 1]>>),
    Poly(PolyConnIteratorMut<'a>),
}

impl<'a> Iterator for ConnectivityIteratorMut<'a> {
    type Item = ArrayViewMut1<'a, usize>;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            ConnectivityIteratorMut::Regular(iter) => iter.next(),
            ConnectivityIteratorMut::Poly(iter) => iter.next(),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        match self {
            ConnectivityIteratorMut::Regular(iter) => iter.size_hint(),
            ConnectivityIteratorMut::Poly(iter) => iter.size_hint(),
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

    pub fn iter(&self) -> impl Iterator<Item = ArrayView1<'_, usize>> + '_ {
        match self {
            Connectivity::Regular(conn) => ConnectivityIterator::Regular(conn.axis_iter(Axis(0))),
            Connectivity::Poly { data, offsets } => ConnectivityIterator::Poly(PolyConnIterator {
                data,
                offsets,
                index: 0,
            }),
        }
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = ArrayViewMut1<'_, usize>> + '_ {
        match self {
            Connectivity::Regular(conn) => {
                ConnectivityIteratorMut::Regular(conn.axis_iter_mut(Axis(0)))
            }
            Connectivity::Poly { data, offsets } => {
                ConnectivityIteratorMut::Poly(PolyConnIteratorMut {
                    data: data.as_slice_mut().unwrap(),
                    offsets: offsets.as_slice().unwrap(),
                    index: 0,
                })
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr1, arr2};

    #[test]
    fn test_connectivity() {
        let conn = arr2(&[[0, 1], [1, 2], [2, 3]]);
        let connectivity = Connectivity::new_regular(conn);
        assert_eq!(connectivity.len(), 3);
        assert_eq!(connectivity.get(0).to_vec(), vec![0, 1]);
        assert_eq!(connectivity.get(1).to_vec(), vec![1, 2]);
        assert_eq!(connectivity.get(2).to_vec(), vec![2, 3]);
    }
    #[test]
    fn test_poly_connectivity() {
        let data = arr1(&[0, 1, 2, 3, 4, 5]);
        let offsets = arr1(&[0, 2, 5, 6]);
        let connectivity = Connectivity::new_poly(data, offsets);
        assert_eq!(connectivity.len(), 3);
        assert_eq!(connectivity.get(0).to_vec(), vec![0, 1]);
        assert_eq!(connectivity.get(1).to_vec(), vec![2, 3, 4]);
        assert_eq!(connectivity.get(2).to_vec(), vec![5]);
    }
    #[test]
    fn test_poly_connectivity_mut() {
        let data = arr1(&[0, 1, 2, 3, 4, 5]);
        let offsets = arr1(&[0, 2, 5, 6]);
        let mut connectivity = Connectivity::new_poly(data.clone(), offsets.clone());
        assert_eq!(connectivity.get_mut(0).to_vec(), vec![0, 1]);
        connectivity.get_mut(1)[0] = 10;
        assert_eq!(connectivity.get_mut(1).to_vec(), vec![10, 3, 4]);
        assert_eq!(connectivity.get_mut(2).to_vec(), vec![5]);
    }
    #[test]
    fn test_poly_connectivity_iter() {
        let data = arr1(&[0, 1, 2, 3, 4, 5]);
        let offsets = arr1(&[0, 2, 5, 6]);
        let connectivity = Connectivity::new_poly(data, offsets);
        let mut iter = connectivity.iter();
        assert_eq!(iter.next().unwrap().to_vec(), vec![0, 1]);
        assert_eq!(iter.next().unwrap().to_vec(), vec![2, 3, 4]);
        assert_eq!(iter.next().unwrap().to_vec(), vec![5]);
    }
    #[test]
    fn test_poly_connectivity_iter_mut() {
        let data = arr1(&[0, 1, 2, 3, 4, 5]);
        let offsets = arr1(&[0, 2, 5, 6]);
        let mut connectivity = Connectivity::new_poly(data.clone(), offsets.clone());
        let mut iter = connectivity.iter_mut();
        assert_eq!(iter.next().unwrap().to_vec(), vec![0, 1]);
        assert_eq!(iter.next().unwrap().to_vec(), vec![2, 3, 4]);
        assert_eq!(iter.next().unwrap().to_vec(), vec![5]);
    }
    #[test]
    fn test_poly_connectivity_iter_size_hint() {
        let data = arr1(&[0, 1, 2, 3, 4, 5]);
        let offsets = arr1(&[0, 2, 5, 6]);
        let connectivity = Connectivity::new_poly(data, offsets);
        let iter = connectivity.iter();
        assert_eq!(iter.size_hint(), (3, Some(3)));
    }
    #[test]
    fn test_poly_connectivity_iter_mut_size_hint() {
        let data = arr1(&[0, 1, 2, 3, 4, 5]);
        let offsets = arr1(&[0, 2, 5, 6]);
        let mut connectivity = Connectivity::new_poly(data.clone(), offsets.clone());
        let iter = connectivity.iter_mut();
        assert_eq!(iter.size_hint(), (3, Some(3)));
    }
}

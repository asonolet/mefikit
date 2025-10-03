use ndarray as nd;
// use rayon::prelude::*;
use super::dataarray::DataArray;
use serde::{Deserialize, Serialize};

// /// Connectivity structure to represent the connectivity of a mesh.
// ///
// /// It can be either regular or polygonal. Regular connectivity is represented as a 2D array,
// /// while polygonal connectivity is represented as a 1D array with offsets. The offsets array
// /// indicates the start and end of each polygon in the data array. The data array contains the
// /// indices of the vertices of the polygons.
// #[derive_where(Clone; C: nd::RawDataClone)]
// #[derive_where(Debug, Serialize, PartialEq, Eq, Hash)]
// #[derive_where(Deserialize; C: nd::DataOwned)]
// pub enum ConnectivityBase<C>
// where
//     C: nd::RawData<Elem = usize> + nd::Data,
// {
//     Regular(nd::ArrayBase<C, nd::Ix2>),
//     Poly {
//         data: nd::ArrayBase<C, nd::Ix1>,
//         offsets: nd::ArrayBase<C, nd::Ix1>,
//     },
// }
//

#[derive(Clone, Debug, Serialize, PartialEq, Eq, Hash)]
pub enum ConnectivityView<'a> {
    Regular(DataArray<'a, usize, nd::Ix2>),
    Poly {
        data: DataArray<'a, usize, nd::Ix1>,
        offsets: DataArray<'a, usize, nd::Ix1>,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum Connectivity {
    Regular(nd::ArcArray<usize, nd::Ix2>),
    Poly {
        data: nd::ArcArray<usize, nd::Ix1>,
        offsets: nd::ArcArray<usize, nd::Ix1>,
    },
}

pub struct ConnectivityIterator<'a> {
    connectivity: ConnectivityView<'a>,
    index: usize,
}

impl<'a> Iterator for ConnectivityIterator<'a> {
    type Item = &'a [usize];

    fn next(&mut self) -> Option<Self::Item> {
        match &self.connectivity {
            ConnectivityView::Regular(arr) => {
                let arr = arr.view();
                if self.index >= arr.shape()[0] {
                    None
                } else {
                    // Why I am not succeeding into using ndarray primitives ???
                    let start = self.index * arr.shape()[1];
                    let end = (self.index + 1) * arr.shape()[1];
                    self.index += 1;
                    Some(&arr.to_slice().unwrap()[start..end])
                }
            }
            ConnectivityView::Poly { data, offsets } => {
                let offsets = offsets.view();
                if self.index >= offsets.len() {
                    return None;
                }
                let start = if self.index == 0 {
                    0
                } else {
                    offsets[self.index - 1]
                };
                let end = offsets[self.index];
                self.index += 1;
                let result = &data.as_slice().unwrap()[start..end];
                Some(result)
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        match self.connectivity {
            ConnectivityView::Regular(arr) => {
                let arr = arr.to_view();
                let remaning_len = arr.shape()[0] - self.index;
                (remaning_len, Some(remaning_len))
            }
            ConnectivityView::Poly { offsets, .. } => {
                let offsets = offsets.to_view();
                let remaning_len = offsets.len() - self.index;
                (remaning_len, Some(remaning_len))
            }
        }
    }
}

impl<'a> ExactSizeIterator for ConnectivityIterator<'a> {}

// pub struct PolyConnIteratorMut<'a> {
//     data: &'a mut [usize],
//     offsets: &'a [usize],
//     index: usize,
// }
//
// impl<'a> Iterator for PolyConnIteratorMut<'a> {
//     type Item = nd::ArrayViewMut1<'a, usize>;
//
//     fn next(&mut self) -> Option<Self::Item> {
//         if self.index >= self.offsets.len() {
//             return None;
//         }
//         let start = if self.index == 0 {
//             0
//         } else {
//             self.offsets[self.index - 1]
//         };
//         let end = self.offsets[self.index];
//         let len = end - start;
//         self.index += 1;
//
//         // Split off the first `len` elements from `self.data`
//         // using `split_at_mut`
//         // Then replace the data slice with the rest of the data.
//         let data = std::mem::take(&mut self.data);
//
//         let (chunk, rest) = data.split_at_mut(len);
//
//         // Update the data slice to point to the rest of the data
//         self.data = rest;
//         // Return the chunk
//         // The chunk is a mutable slice of the original data
//         // and the rest is the remaining data
//         Some(nd::ArrayViewMut1::from(chunk))
//     }
//
//     fn size_hint(&self) -> (usize, Option<usize>) {
//         let len = self.offsets.len();
//         (len, Some(len))
//     }
// }
//
// enum ConnectivityIteratorMut<'a> {
//     Regular(),
//     Poly(PolyConnIteratorMut<'a>),
// }
//
// impl<'a> Iterator for ConnectivityIteratorMut<'a> {
//     type Item = nd::ArrayViewMut1<'a, usize>;
//
//     fn next(&mut self) -> Option<Self::Item> {
//         match self {
//             ConnectivityIteratorMut::Regular(iter) => iter.next(),
//             ConnectivityIteratorMut::Poly(iter) => iter.next(),
//         }
//     }
//
//     fn size_hint(&self) -> (usize, Option<usize>) {
//         match self {
//             ConnectivityIteratorMut::Regular(iter) => iter.size_hint(),
//             ConnectivityIteratorMut::Poly(iter) => iter.size_hint(),
//         }
//     }
// }

impl Connectivity {
    pub fn new_regular(conn: nd::Array2<usize>) -> Self {
        Connectivity::Regular(conn)
    }

    pub fn new_poly(data: nd::Array1<usize>, offsets: nd::Array1<usize>) -> Self {
        Connectivity::Poly { data, offsets }
    }

    pub fn append(&mut self, connectivity: nd::ArrayView1<usize>) {
        match self {
            Connectivity::Regular(conn) => {
                conn.push_row(connectivity).unwrap();
            }
            Connectivity::Poly { data, offsets } => {
                data.append(nd::Axis(0), connectivity).unwrap();
                offsets
                    .append(nd::Axis(0), nd::arr1(&[data.len()]).view())
                    .unwrap();
            }
        }
    }
}

impl<'a> ConnectivityView<'a> {
    pub fn len(&'a self) -> usize {
        match self {
            ConnectivityView::Regular(conn) => conn.view().nrows(),
            ConnectivityView::Poly { offsets, .. } => offsets.view().len(),
        }
    }

    pub fn get(&'a self, index: usize) -> &'a [usize] {
        match self {
            ConnectivityView::Regular(conn) => conn.row_slice(index),
            ConnectivityView::Poly { data, offsets } => {
                let offsets = offsets.view();
                let start = if index == 0 { 0 } else { offsets[index - 1] };
                let end = offsets[index];
                &data.as_slice().unwrap()[start..end]
            }
        }
    }

    pub fn iter(&'a self) -> impl ExactSizeIterator<Item = &'a [usize]> + 'a {
        match self {
            ConnectivityView::Regular(arr) => {
                (0..arr.len()).map(move |i| arr.view().row(i).as_slice().unwrap())
            }
            ConnectivityView::Poly { data, offsets } => {
                (0..offsets.len()).map(move |i| {
                    let offsets = offsets.view();
                    let start = if i == 0 { 0 } else { offsets[i - 1] };
                    let end = offsets[i];
                    &data.as_slice().unwrap()[start..end]
                })
                // let data = data.view();
                // let offsets = offsets.view();
                // if self.index >= offsets.len() {
                //     return None;
                // }
                // let start = if self.index == 0 {
                //     0
                // } else {
                //     offsets[self.index - 1]
                // };
                // let end = offsets[self.index];
                // self.index += 1;
                // let result = &data.to_slice().unwrap()[start..end];
                // Some(result)
            }
        }
    }

    // pub fn get_mut(&mut self, index: usize) -> &mut [usize]
    // where
    //     C: nd::DataMut,
    // {
    //     match self {
    //         ConnectivityBase::Regular(conn) => conn.row_mut(index).as_slice_mut().unwrap(),
    //         ConnectivityBase::Poly { data, offsets } => {
    //             let start = if index == 0 { 0 } else { offsets[index - 1] };
    //             let end = offsets[index];
    //             &mut data.as_slice_mut().unwrap()[start..end]
    //         }
    //     }
    // }

    // pub fn iter_mut(&mut self) -> impl Iterator<Item = nd::ArrayViewMut1<'_, usize>> + '_
    // where
    //     C: nd::DataMut,
    // {
    //     match self {
    //         ConnectivityBase::Regular(conn) => {
    //             ConnectivityIteratorMut::Regular(conn.axis_iter_mut(nd::Axis(0)))
    //         }
    //         ConnectivityBase::Poly { data, offsets } => {
    //             ConnectivityIteratorMut::Poly(PolyConnIteratorMut {
    //                 data: data.as_slice_mut().unwrap(),
    //                 offsets: offsets.as_slice().unwrap(),
    //                 index: 0,
    //             })
    //         }
    //     }
    // }
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
        let offsets = arr1(&[2, 5, 6]);
        let connectivity = Connectivity::new_poly(data, offsets);
        assert_eq!(connectivity.len(), 3);
        assert_eq!(connectivity.get(0).to_vec(), vec![0, 1]);
        assert_eq!(connectivity.get(1).to_vec(), vec![2, 3, 4]);
        assert_eq!(connectivity.get(2).to_vec(), vec![5]);
    }
    // #[test]
    // fn test_poly_connectivity_mut() {
    //     let data = arr1(&[0, 1, 2, 3, 4, 5]);
    //     let offsets = arr1(&[2, 5, 6]);
    //     let mut connectivity = Connectivity::new_poly(data.clone(), offsets.clone());
    //     assert_eq!(connectivity.get_mut(0).to_vec(), vec![0, 1]);
    //     connectivity.get_mut(1)[0] = 10;
    //     assert_eq!(connectivity.get_mut(1).to_vec(), vec![10, 3, 4]);
    //     assert_eq!(connectivity.get_mut(2).to_vec(), vec![5]);
    // }
    #[test]
    fn test_poly_connectivity_iter() {
        let data = arr1(&[0, 1, 2, 3, 4, 5]);
        let offsets = arr1(&[2, 5, 6]);
        let connectivity = Connectivity::new_poly(data, offsets);
        let mut iter = connectivity.iter();
        assert_eq!(iter.next().unwrap().to_vec(), vec![0, 1]);
        assert_eq!(iter.next().unwrap().to_vec(), vec![2, 3, 4]);
        assert_eq!(iter.next().unwrap().to_vec(), vec![5]);
    }
    // #[test]
    // fn test_poly_connectivity_iter_mut() {
    //     let data = arr1(&[0, 1, 2, 3, 4, 5]);
    //     let offsets = arr1(&[2, 5, 6]);
    //     let mut connectivity = Connectivity::new_poly(data.clone(), offsets.clone());
    //     let mut iter = connectivity.iter_mut();
    //     assert_eq!(iter.next().unwrap().to_vec(), vec![0, 1]);
    //     assert_eq!(iter.next().unwrap().to_vec(), vec![2, 3, 4]);
    //     assert_eq!(iter.next().unwrap().to_vec(), vec![5]);
    // }
    #[test]
    fn test_poly_connectivity_iter_size_hint() {
        let data = arr1(&[0, 1, 2, 3, 4, 5]);
        let offsets = arr1(&[2, 5, 6]);
        let connectivity = Connectivity::new_poly(data, offsets);
        let iter = connectivity.iter();
        assert_eq!(iter.size_hint(), (3, Some(3)));
    }
    // #[test]
    // fn test_poly_connectivity_iter_mut_size_hint() {
    //     let data = arr1(&[0, 1, 2, 3, 4, 5]);
    //     let offsets = arr1(&[2, 5, 6]);
    //     let mut connectivity = Connectivity::new_poly(data.clone(), offsets.clone());
    //     let iter = connectivity.iter_mut();
    //     assert_eq!(iter.size_hint(), (3, Some(3)));
    // }
}

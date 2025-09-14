use derive_where::derive_where;
use ndarray as nd;
// use rayon::prelude::*;

/// Connectivity structure to represent the connectivity of a mesh.
///
/// It can be either regular or polygonal. Regular connectivity is represented as a 2D array,
/// while polygonal connectivity is represented as a 1D array with offsets. The offsets array
/// indicates the start and end of each polygon in the data array. The data array contains the
/// indices of the vertices of the polygons.
#[derive_where(Clone; C: nd::RawDataClone)]
#[derive_where(Debug, Serialize, PartialEq, Eq, Hash)]
#[derive_where(Deserialize; C: nd::DataOwned)]
pub enum ConnectivityBase<C>
where
    C: nd::RawData<Elem = usize> + nd::Data,
{
    Regular(nd::ArrayBase<C, nd::Ix2>),
    Poly {
        data: nd::ArrayBase<C, nd::Ix1>,
        offsets: nd::ArrayBase<C, nd::Ix1>,
    },
}

pub type Connectivity = ConnectivityBase<nd::OwnedRepr<usize>>;
pub type ConnectivityView<'a> = ConnectivityBase<nd::ViewRepr<&'a usize>>;

pub struct ConnectivityIterator<'a> {
    connectivity: ConnectivityView<'a>,
    index: usize,
}

// pub struct PolyConnIterator<'a> {
//     data: &'a [usize],
//     offsets: &'a [usize],
//     index: usize,
// }
//
// impl<'a> Iterator for PolyConnIterator<'a> {
//     type Item = &'a[usize];
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
//         self.index += 1;
//         let result = &self.data[start..end];
//         Some(result)
//     }
//
//     fn size_hint(&self) -> (usize, Option<usize>) {
//         let len = self.offsets.len();
//         (len, Some(len))
//     }
// }

impl<'a> Iterator for ConnectivityIterator<'a> {
    type Item = &'a [usize];

    fn next(&mut self) -> Option<Self::Item> {
        match self.connectivity {
            ConnectivityView::Regular(arr) => {
                if self.index >= arr.shape()[0] {
                    None
                } else {
                    // Why I am not succinding into using ndarray primitives ???
                    let start = self.index * arr.shape()[1];
                    let end = (self.index + 1) * arr.shape()[1];
                    self.index += 1;
                    Some(&arr.to_slice().unwrap()[start..end])
                }
            }
            ConnectivityView::Poly { data, offsets } => {
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
                let result = &data.to_slice().unwrap()[start..end];
                Some(result)
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        match self.connectivity {
            ConnectivityView::Regular(arr) => {
                let len = arr.shape()[0];
                (len, Some(len))
            }
            ConnectivityView::Poly { offsets, .. } => {
                let len = offsets.len();
                (len, Some(len))
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

impl<C> ConnectivityBase<C>
where
    C: nd::RawData<Elem = usize> + nd::Data,
{
    pub fn len(&self) -> usize {
        match self {
            ConnectivityBase::Regular(conn) => conn.nrows(),
            ConnectivityBase::Poly { offsets, .. } => offsets.len(),
        }
    }

    pub fn get(&self, index: usize) -> &[usize]
    where
        C: nd::Data,
    {
        match self {
            ConnectivityBase::Regular(conn) => conn.row(index).to_slice().unwrap(),
            ConnectivityBase::Poly { data, offsets } => {
                let start = if index == 0 { 0 } else { offsets[index - 1] };
                let end = offsets[index];
                &data.as_slice().unwrap()[start..end]
            }
        }
    }
    pub fn iter(&self) -> impl Iterator<Item = &'_ [usize]> + '_
    where
        C: nd::Data,
    {
        ConnectivityIterator {
            connectivity: self.view(),
            index: 0,
        }
    }

    pub fn view(&self) -> ConnectivityView {
        match self {
            ConnectivityBase::Regular(arr) => ConnectivityView::Regular(arr.view()),
            ConnectivityBase::Poly { data, offsets } => ConnectivityView::Poly {
                data: data.view(),
                offsets: offsets.view(),
            },
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

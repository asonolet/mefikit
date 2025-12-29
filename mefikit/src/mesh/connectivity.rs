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

pub type Connectivity = ConnectivityBase<nd::OwnedArcRepr<usize>>;
pub type ConnectivityView<'a> = ConnectivityBase<nd::ViewRepr<&'a usize>>;

pub struct ConnectivityIterator<'a> {
    connectivity: ConnectivityView<'a>,
    index: usize,
}

impl<'a> Iterator for ConnectivityIterator<'a> {
    type Item = &'a [usize];

    fn next(&mut self) -> Option<Self::Item> {
        match self.connectivity {
            ConnectivityView::Regular(arr) => {
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
                let remaning_len = arr.shape()[0] - self.index;
                (remaning_len, Some(remaning_len))
            }
            ConnectivityView::Poly { offsets, .. } => {
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
    pub fn new_regular(conn: nd::ArcArray2<usize>) -> Self {
        Connectivity::Regular(conn)
    }

    pub fn new_poly(data: nd::ArcArray1<usize>, offsets: nd::ArcArray1<usize>) -> Self {
        Connectivity::Poly { data, offsets }
    }

    pub fn append(&mut self, connectivity: nd::ArrayView1<usize>) {
        match self {
            Connectivity::Regular(conn) => {
                let mut conn = std::mem::take(conn).into_owned();
                conn.push_row(connectivity).unwrap();
                let _ = std::mem::replace(self, Connectivity::Regular(conn.into_shared()));
            }
            Connectivity::Poly { data, offsets } => {
                let mut data = std::mem::take(data).into_owned();
                let mut offsets = std::mem::take(offsets).into_owned();
                data.append(nd::Axis(0), connectivity).unwrap();
                offsets
                    .append(nd::Axis(0), nd::arr1(&[data.len()]).view())
                    .unwrap();
                let _ = std::mem::replace(
                    self,
                    Connectivity::Poly {
                        data: data.into_shared(),
                        offsets: offsets.into_shared(),
                    },
                );
            }
        }
    }

    pub fn get_mut(&mut self, index: usize) -> &mut [usize] {
        match self {
            ConnectivityBase::Regular(conn) => {
                let start = conn.shape()[1] * index;
                let end = conn.shape()[1] * (index + 1);
                &mut conn.as_slice_mut().unwrap()[start..end]
            }
            ConnectivityBase::Poly { data, offsets } => {
                let start = if index == 0 { 0 } else { offsets[index - 1] };
                let end = offsets[index];
                &mut data.as_slice_mut().unwrap()[start..end]
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
    pub fn iter(&self) -> impl ExactSizeIterator<Item = &'_ [usize]> + '_
    where
        C: nd::Data,
    {
        ConnectivityIterator {
            connectivity: self.view(),
            index: 0,
        }
    }

    pub fn view(&self) -> ConnectivityView<'_> {
        match self {
            ConnectivityBase::Regular(arr) => ConnectivityView::Regular(arr.view()),
            ConnectivityBase::Poly { data, offsets } => ConnectivityView::Poly {
                data: data.view(),
                offsets: offsets.view(),
            },
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
        let connectivity = Connectivity::new_regular(conn.into_shared());
        assert_eq!(connectivity.len(), 3);
        assert_eq!(connectivity.get(0).to_vec(), vec![0, 1]);
        assert_eq!(connectivity.get(1).to_vec(), vec![1, 2]);
        assert_eq!(connectivity.get(2).to_vec(), vec![2, 3]);
    }
    #[test]
    fn test_poly_connectivity() {
        let data = arr1(&[0, 1, 2, 3, 4, 5]);
        let offsets = arr1(&[2, 5, 6]);
        let connectivity = Connectivity::new_poly(data.into_shared(), offsets.into_shared());
        assert_eq!(connectivity.len(), 3);
        assert_eq!(connectivity.get(0).to_vec(), vec![0, 1]);
        assert_eq!(connectivity.get(1).to_vec(), vec![2, 3, 4]);
        assert_eq!(connectivity.get(2).to_vec(), vec![5]);
    }
    #[test]
    fn test_poly_connectivity_iter() {
        let data = arr1(&[0, 1, 2, 3, 4, 5]).into_shared();
        let offsets = arr1(&[2, 5, 6]).into_shared();
        let connectivity = Connectivity::new_poly(data, offsets);
        let mut iter = connectivity.iter();
        assert_eq!(iter.next().unwrap().to_vec(), vec![0, 1]);
        assert_eq!(iter.next().unwrap().to_vec(), vec![2, 3, 4]);
        assert_eq!(iter.next().unwrap().to_vec(), vec![5]);
    }
    #[test]
    fn test_poly_connectivity_iter_size_hint() {
        let data = arr1(&[0, 1, 2, 3, 4, 5]).into_shared();
        let offsets = arr1(&[2, 5, 6]).into_shared();
        let connectivity = Connectivity::new_poly(data, offsets);
        let iter = connectivity.iter();
        assert_eq!(iter.size_hint(), (3, Some(3)));
    }
    #[test]
    fn test_append_regular() {
        let conn = arr2(&[[0, 1], [1, 2]]);
        let mut connectivity = Connectivity::new_regular(conn.into_shared());
        connectivity.append(arr1(&[2, 3]).view());
        assert_eq!(connectivity.len(), 3);
        assert_eq!(connectivity.get(2).to_vec(), vec![2, 3]);
    }
    #[test]
    fn test_append_poly() {
        let data = arr1(&[0, 1, 2, 3]);
        let offsets = arr1(&[2, 4]);
        let mut connectivity = Connectivity::new_poly(data.into_shared(), offsets.into_shared());
        connectivity.append(arr1(&[4, 5, 6]).view());
        assert_eq!(connectivity.len(), 3);
        assert_eq!(connectivity.get(2).to_vec(), vec![4, 5, 6]);
    }
    #[test]
    fn test_regular_iter() {
        let conn = arr2(&[[0, 1], [1, 2], [2, 3]]);
        let connectivity = Connectivity::new_regular(conn.into_shared());
        let mut iter = connectivity.iter();
        assert_eq!(iter.next().unwrap().to_vec(), vec![0, 1]);
        assert_eq!(iter.next().unwrap().to_vec(), vec![1, 2]);
        assert_eq!(iter.next().unwrap().to_vec(), vec![2, 3]);
    }
    #[test]
    fn test_regular_iter_size_hint() {
        let conn = arr2(&[[0, 1], [1, 2], [2, 3]]);
        let connectivity = Connectivity::new_regular(conn.into_shared());
        let iter = connectivity.iter();
        assert_eq!(iter.size_hint(), (3, Some(3)));
    }
    #[test]
    fn test_connectivity_get_out_of_bounds() {
        let conn = arr2(&[[0, 1], [1, 2], [2, 3]]);
        let connectivity = Connectivity::new_regular(conn.into_shared());
        let result = std::panic::catch_unwind(|| {
            connectivity.get(3);
        });
        assert!(result.is_err());
    }
}

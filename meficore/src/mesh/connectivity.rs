use std::ops::{Index, IndexMut};

use derive_where::derive_where;
use ndarray as nd;

use crate::mesh::indirect_index::{IndirectIndex, IndirectIndexIter};
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
    Poly(IndirectIndex<usize, C, C>),
}

pub type Connectivity = ConnectivityBase<nd::OwnedArcRepr<usize>>;
pub type ConnectivityView<'a> = ConnectivityBase<nd::ViewRepr<&'a usize>>;

pub enum ConnectivityIterator<'a> {
    RegularIter(nd::iter::LanesIter<'a, usize, nd::Ix1>),
    PolyIter(IndirectIndexIter<'a, usize>),
}

impl<'a> Iterator for ConnectivityIterator<'a> {
    type Item = &'a [usize];
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            ConnectivityIterator::RegularIter(it) => it.next().map(|f| f.to_slice().unwrap()),
            ConnectivityIterator::PolyIter(it) => it.next(),
        }
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        match self {
            ConnectivityIterator::RegularIter(it) => it.size_hint(),
            ConnectivityIterator::PolyIter(it) => it.size_hint(),
        }
    }
}

impl<'a> ExactSizeIterator for ConnectivityIterator<'a> {}

impl Connectivity {
    pub fn new_regular(conn: nd::ArcArray2<usize>) -> Self {
        Connectivity::Regular(conn)
    }

    pub fn new_poly(data: nd::ArcArray1<usize>, offsets: nd::ArcArray1<usize>) -> Self {
        Connectivity::Poly(IndirectIndex { data, offsets })
    }

    pub fn push(&mut self, connectivity: nd::ArrayView1<usize>) {
        match self {
            Connectivity::Regular(conn) => {
                let mut conn = std::mem::take(conn).into_owned();
                conn.push_row(connectivity).unwrap();
                let _ = std::mem::replace(self, Connectivity::Regular(conn.into_shared()));
            }
            Connectivity::Poly(conn) => {
                let conn = std::mem::take(conn);
                let mut conn = conn.into_owned();
                conn.push_conn(connectivity);
                let _ = std::mem::replace(self, Connectivity::Poly(conn.into_shared()));
            }
        }
    }
}

impl IndexMut<usize> for Connectivity {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        match self {
            ConnectivityBase::Regular(conn) => {
                let start = conn.shape()[1] * index;
                let end = conn.shape()[1] * (index + 1);
                &mut conn.as_slice_mut().unwrap()[start..end]
            }
            ConnectivityBase::Poly(conn) => conn.index_mut(index),
        }
    }
}

impl<C> ConnectivityBase<C>
where
    C: nd::Data<Elem = usize>,
{
    pub fn len(&self) -> usize {
        match self {
            ConnectivityBase::Regular(conn) => conn.nrows(),
            ConnectivityBase::Poly(conn) => conn.len(),
        }
    }

    pub fn iter(&self) -> impl ExactSizeIterator<Item = &'_ [usize]> + '_
    where
        C: nd::Data,
    {
        match self {
            Self::Regular(conn) => ConnectivityIterator::RegularIter(conn.rows().into_iter()),
            Self::Poly(conn) => ConnectivityIterator::PolyIter(conn.iter()),
        }
    }

    pub fn view(&self) -> ConnectivityView<'_> {
        match self {
            Self::Regular(arr) => ConnectivityView::Regular(arr.view()),
            Self::Poly(conn) => ConnectivityView::Poly(conn.view()),
        }
    }
}

impl<C> Index<usize> for ConnectivityBase<C>
where
    C: nd::RawData<Elem = usize> + nd::Data,
{
    type Output = [usize];

    fn index(&self, index: usize) -> &Self::Output {
        match self {
            Self::Regular(conn) => conn.row(index).to_slice().unwrap(),
            Self::Poly(conn) => &conn[index],
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
        assert_eq!(connectivity[0].to_vec(), vec![0, 1]);
        assert_eq!(connectivity[1].to_vec(), vec![1, 2]);
        assert_eq!(connectivity[2].to_vec(), vec![2, 3]);
    }
    #[test]
    fn test_poly_connectivity() {
        let data = arr1(&[0, 1, 2, 3, 4, 5]);
        let offsets = arr1(&[2, 5, 6]);
        let connectivity = Connectivity::new_poly(data.into_shared(), offsets.into_shared());
        assert_eq!(connectivity.len(), 3);
        assert_eq!(connectivity[0].to_vec(), vec![0, 1]);
        assert_eq!(connectivity[1].to_vec(), vec![2, 3, 4]);
        assert_eq!(connectivity[2].to_vec(), vec![5]);
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
        connectivity.push(arr1(&[2, 3]).view());
        assert_eq!(connectivity.len(), 3);
        assert_eq!(connectivity[2].to_vec(), vec![2, 3]);
    }
    #[test]
    fn test_append_poly() {
        let data = arr1(&[0, 1, 2, 3]);
        let offsets = arr1(&[2, 4]);
        let mut connectivity = Connectivity::new_poly(data.into_shared(), offsets.into_shared());
        connectivity.push(arr1(&[4, 5, 6]).view());
        assert_eq!(connectivity.len(), 3);
        assert_eq!(connectivity[2].to_vec(), vec![4, 5, 6]);
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
            &connectivity[3];
        });
        assert!(result.is_err());
    }
}

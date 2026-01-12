use std::collections::VecDeque;

use derive_where::derive_where;
use ndarray as nd;
use serde::Serialize;

#[derive_where(Clone; C: nd::RawDataClone, D: nd::RawDataClone)]
#[derive_where(Debug, Serialize, PartialEq, Hash)]
#[derive_where(Eq; T: Eq)]
// #[derive_where(Deserialize; C: nd::DataOwned, D: nd::DataOwned, T: 'static + Deserialize)]
pub struct IndirectIndex<T, C, D>
where
    T: Clone + Serialize + PartialEq + std::fmt::Debug + std::hash::Hash,
    C: nd::RawData<Elem = T> + nd::Data,
    D: nd::RawData<Elem = usize> + nd::Data,
{
    data: nd::ArrayBase<C, nd::Ix1>,
    offsets: nd::ArrayBase<D, nd::Ix1>,
}

impl<T, C, D> IndirectIndex<T, C, D>
where
    T: Clone + Serialize + PartialEq + std::fmt::Debug + std::hash::Hash,
    C: nd::RawData<Elem = T> + nd::Data,
    D: nd::RawData<Elem = usize> + nd::Data,
{
    fn get(&self, i: usize) -> &[T] {
        let start = match i {
            0 => 0,
            i => self.offsets[i - 1],
        };
        let stop = self.offsets[i];
        &self.data.as_slice().unwrap()[start..stop]
    }
    fn iter(&self) -> IndirectIndexIter<'_, T> {
        IndirectIndexIter {
            data: self.data.as_slice().unwrap(),
            offsets: self.offsets.as_slice().unwrap(),
            last_offset: 0,
        }
    }
}

impl<T, C, D> IndirectIndex<T, C, D>
where
    T: Clone + Serialize + PartialEq + std::fmt::Debug + std::hash::Hash,
    C: nd::RawDataClone<Elem = T> + nd::DataOwned + nd::DataMut,
    D: nd::RawDataClone<Elem = usize> + nd::DataOwned + nd::DataMut,
{
    fn get_mut(&mut self, i: usize) -> &mut [T] {
        let start = match i {
            0 => 0,
            i => self.offsets[i - 1],
        };
        let stop = self.offsets[i];
        &mut self.data.as_slice_mut().unwrap()[start..stop]
    }
    fn iter_mut(&mut self) -> IndirectIndexIterMut<'_, T> {
        IndirectIndexIterMut {
            data: self.data.as_slice_mut().unwrap(),
            offsets: self.offsets.as_slice_mut().unwrap(),
            last_offset: 0,
        }
    }
    // fn push(&mut self, elem: &[T])
    // where
    //     T: Clone,
    // {
    //     let data = std::mem::replace(&mut self.data, nd::arr1(&[]));
    //     self.data.extend(elem.iter().cloned());
    //     self.offsets.push(self.data.len());
    // }
    // fn extend_from_slice(&mut self, data: &[T], offsets: &[usize])
    // where
    //     T: Clone,
    // {
    //     let num_elems = self.data.len();
    //     self.data.extend_from_slice(data);
    //     self.offsets.extend(offsets.iter().map(|o| o + num_elems));
    // }
}

// impl<'a, T> Extend<&'a [T]> for IndirectIndex<T>
// where
//     T: Clone,
// {
//     fn extend<I>(&mut self, iter: I)
//     where
//         I: IntoIterator<Item = &'a [T]>,
//     {
//         let num_elems = self.data.len();
//         let (data, offsets): (Vec<&[T]>, Vec<usize>) =
//             iter.into_iter().map(|sl| (sl, sl.len())).collect();
//         self.data.extend(data.into_iter().flatten().cloned());
//         self.offsets.extend(offsets.iter().map(|o| o + num_elems));
//     }
// }

pub struct IndirectIndexIter<'a, T>
where
    T: 'a,
{
    data: &'a [T],
    offsets: &'a [usize],
    last_offset: usize,
}

impl<'a, T> Iterator for IndirectIndexIter<'a, T> {
    type Item = &'a [T];

    fn next(&mut self) -> Option<Self::Item> {
        if self.offsets.is_empty() {
            return None;
        }
        let offset = self.offsets.split_off_first().unwrap();
        let len_elem = *offset - self.last_offset;
        self.last_offset = *offset;
        let data = std::mem::take(&mut self.data);
        let (chunk, rest) = data.split_at(len_elem);
        self.data = rest;
        Some(chunk)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.offsets.len(), Some(self.offsets.len()))
    }
}

impl<'a, T, C, D> IntoIterator for &'a IndirectIndex<T, C, D>
where
    T: Clone + Serialize + PartialEq + std::fmt::Debug + std::hash::Hash,
    C: nd::RawDataClone<Elem = T> + nd::Data,
    D: nd::RawDataClone<Elem = usize> + nd::Data,
{
    type Item = &'a [T];
    type IntoIter = IndirectIndexIter<'a, T>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

pub struct IndirectIndexIterMut<'a, T>
where
    T: 'a,
{
    data: &'a mut [T],
    offsets: &'a mut [usize],
    last_offset: usize,
}

impl<'a, T> Iterator for IndirectIndexIterMut<'a, T>
where
    T: Clone,
{
    type Item = &'a mut [T];

    fn next(&mut self) -> Option<Self::Item> {
        if self.offsets.is_empty() {
            return None;
        }
        let offset = self.offsets.split_off_first_mut().unwrap();
        let len_elem = *offset - self.last_offset;
        self.last_offset = *offset;
        let data = std::mem::take(&mut self.data);
        let (chunk, rest) = data.split_at_mut(len_elem);
        self.data = rest;
        Some(chunk)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.offsets.len(), Some(self.offsets.len()))
    }
}

impl<'a, T, C, D> IntoIterator for &'a mut IndirectIndex<T, C, D>
where
    T: Clone + Serialize + PartialEq + std::fmt::Debug + std::hash::Hash,
    C: nd::RawDataClone<Elem = T> + nd::DataOwned + nd::DataMut,
    D: nd::RawDataClone<Elem = usize> + nd::DataOwned + nd::DataMut,
{
    type Item = &'a mut [T];
    type IntoIter = IndirectIndexIterMut<'a, T>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

pub struct IndirectIndexIntoIter<T> {
    data: VecDeque<T>,
    offsets: VecDeque<usize>,
    last_offset: usize,
}

impl<T> Iterator for IndirectIndexIntoIter<T> {
    type Item = Vec<T>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.offsets.is_empty() {
            return None;
        }
        let offset = self.offsets.pop_front().unwrap();
        let len_elem = offset - self.last_offset;
        self.last_offset = offset;
        let chunk = self.data.split_off(len_elem);
        Some(chunk.into())
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.offsets.len(), Some(self.offsets.len()))
    }
}

impl<T> IntoIterator for IndirectIndex<T, nd::OwnedRepr<T>, nd::OwnedRepr<usize>>
where
    T: Clone + Serialize + PartialEq + std::fmt::Debug + std::hash::Hash,
{
    type Item = Vec<T>;
    type IntoIter = IndirectIndexIntoIter<T>;
    fn into_iter(self) -> Self::IntoIter {
        let (vec_data, _) = self.data.into_raw_vec_and_offset();
        let (vec_offsets, _) = self.offsets.into_raw_vec_and_offset();
        IndirectIndexIntoIter {
            data: vec_data.into(),
            offsets: vec_offsets.into(),
            last_offset: 0,
        }
    }
}

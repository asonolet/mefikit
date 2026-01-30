use std::{
    collections::VecDeque,
    ops::{Index, IndexMut},
};

use derive_where::derive_where;
use ndarray as nd;
use serde::de::DeserializeOwned;

#[derive_where(Clone; C: nd::RawDataClone<Elem=T>, D: nd::RawDataClone<Elem=usize>, T: Clone)]
#[derive_where(Debug, Serialize, PartialEq, Eq, Hash; T)]
#[derive_where(Deserialize; C: nd::Data<Elem=T> + nd::DataOwned, D: nd::Data<Elem=usize> + nd::DataOwned, T: DeserializeOwned)]
pub struct IndirectIndex<T, C, D>
where
    C: nd::Data<Elem = T>,
    D: nd::Data<Elem = usize>,
{
    pub data: nd::ArrayBase<C, nd::Ix1>,
    pub offsets: nd::ArrayBase<D, nd::Ix1>,
}

impl<T, C, D> IndirectIndex<T, C, D>
where
    C: nd::Data<Elem = T>,
    D: nd::Data<Elem = usize>,
{
    pub fn iter(&self) -> IndirectIndexIter<'_, T> {
        IndirectIndexIter {
            data: self.data.as_slice().unwrap(),
            offsets: self.offsets.as_slice().unwrap(),
            last_offset: 0,
        }
    }
    pub fn len(&self) -> usize {
        self.offsets.len()
    }
    pub fn num_elems_tot(&self) -> usize {
        self.data.len()
    }
    pub fn view(&self) -> IndirectIndexView<'_, T> {
        IndirectIndexView {
            data: self.data.view(),
            offsets: self.offsets.view(),
        }
    }
}

impl<T, C, D> Index<usize> for IndirectIndex<T, C, D>
where
    C: nd::Data<Elem = T>,
    D: nd::Data<Elem = usize>,
{
    type Output = [T];
    fn index(&self, i: usize) -> &Self::Output {
        let start = match i {
            0 => 0,
            i => self.offsets[i - 1],
        };
        let stop = self.offsets[i];
        &self.data.as_slice().unwrap()[start..stop]
    }
}

impl<T, C, D> IndirectIndex<T, C, D>
where
    C: nd::RawDataClone<Elem = T> + nd::DataOwned + nd::DataMut,
    D: nd::RawDataClone<Elem = usize> + nd::DataOwned + nd::DataMut,
{
    pub fn iter_mut(&mut self) -> IndirectIndexIterMut<'_, T> {
        IndirectIndexIterMut {
            data: self.data.as_slice_mut().unwrap(),
            offsets: self.offsets.as_slice_mut().unwrap(),
            last_offset: 0,
        }
    }
}

impl<T, C, D> IndexMut<usize> for IndirectIndex<T, C, D>
where
    C: nd::RawDataClone<Elem = T> + nd::DataOwned + nd::DataMut,
    D: nd::RawDataClone<Elem = usize> + nd::DataOwned + nd::DataMut,
{
    fn index_mut(&mut self, i: usize) -> &mut Self::Output {
        let start = match i {
            0 => 0,
            i => self.offsets[i - 1],
        };
        let stop = self.offsets[i];
        &mut self.data.as_slice_mut().unwrap()[start..stop]
    }
}

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

impl<'a, T> Iterator for IndirectIndexIterMut<'a, T> {
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
    T: Clone,
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

pub type IndirectIndexOwned<T> = IndirectIndex<T, nd::OwnedRepr<T>, nd::OwnedRepr<usize>>;
pub type IndirectIndexShared<T> = IndirectIndex<T, nd::OwnedArcRepr<T>, nd::OwnedArcRepr<usize>>;
pub type IndirectIndexView<'a, T> = IndirectIndex<T, nd::ViewRepr<&'a T>, nd::ViewRepr<&'a usize>>;

impl<T> IndirectIndexOwned<T>
where
    T: Clone,
{
    pub fn new() -> Self {
        Self {
            data: nd::arr1(&[]),
            offsets: nd::arr1(&[]),
        }
    }
    pub fn push(&mut self, elem: &[T]) {
        let data = std::mem::replace(&mut self.data, nd::arr1(&[]));
        let (mut vec_data, _) = data.into_raw_vec_and_offset();
        let offsets = std::mem::replace(&mut self.offsets, nd::arr1(&[]));
        let (mut vec_offsets, _) = offsets.into_raw_vec_and_offset();
        vec_data.extend_from_slice(elem);
        vec_offsets.push(self.data.len());
        self.data = vec_data.into();
        self.offsets = vec_offsets.into();
    }
    pub fn push_conn(&mut self, elem: nd::ArrayView1<'_, T>) {
        self.data.append(nd::Axis(0), elem).unwrap();
        self.offsets
            .push(nd::Axis(0), nd::arr0(self.data.len()).view())
            .unwrap();
    }
    pub fn extend_from_raw_slices(&mut self, data_slice: &[T], offsets_slice: &[usize]) {
        let num_elems = self.data.len();
        let data = std::mem::replace(&mut self.data, nd::arr1(&[]));
        let (mut vec_data, _) = data.into_raw_vec_and_offset();
        let offsets = std::mem::replace(&mut self.offsets, nd::arr1(&[]));
        let (mut vec_offsets, _) = offsets.into_raw_vec_and_offset();
        vec_data.extend_from_slice(data_slice);
        vec_offsets.extend(offsets_slice.iter().map(|of| of + num_elems));
        self.data = vec_data.into();
        self.offsets = vec_offsets.into();
    }
    pub fn reserve(&mut self, additional_data: usize, additional_offsets: usize) {
        self.data.reserve(nd::Axis(0), additional_data).unwrap();
        self.offsets
            .reserve(nd::Axis(0), additional_offsets)
            .unwrap();
    }
    pub fn into_shared(self) -> IndirectIndexShared<T> {
        IndirectIndexShared {
            data: self.data.into_shared(),
            offsets: self.offsets.into_shared(),
        }
    }
}

impl<'a, T> Extend<&'a [T]> for IndirectIndexOwned<T>
where
    T: Clone,
{
    fn extend<I>(&mut self, iter: I)
    where
        I: IntoIterator<Item = &'a [T]>,
    {
        let it = iter.into_iter();
        let (len, _) = it.size_hint();
        self.reserve(2 * len, len);
        for elem in it {
            self.push(elem);
        }
    }
}

impl<T> IntoIterator for IndirectIndexOwned<T>
where
    T: Clone,
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

impl<T> Default for IndirectIndexOwned<T>
where
    T: Clone,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T> IndirectIndexShared<T>
where
    T: Clone,
{
    pub fn new() -> Self {
        Self {
            data: nd::arr1(&[]).into_shared(),
            offsets: nd::arr1(&[]).into_shared(),
        }
    }
    pub fn into_owned(self) -> IndirectIndexOwned<T> {
        IndirectIndexOwned {
            data: self.data.into_owned(),
            offsets: self.offsets.into_owned(),
        }
    }
}

impl<T> Default for IndirectIndexShared<T>
where
    T: Clone,
{
    fn default() -> Self {
        Self::new()
    }
}

use std::collections::VecDeque;

#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct IndirectIndex<T> {
    data: Vec<T>,
    offsets: Vec<usize>,
}

impl<T> IndirectIndex<T> {
    fn get(&self, i: usize) -> &[T] {
        let start = match i {
            0 => 0,
            i => self.offsets[i - 1],
        };
        let stop = self.offsets[i];
        &self.data[start..stop]
    }
    fn get_mut(&mut self, i: usize) -> &mut [T] {
        let start = match i {
            0 => 0,
            i => self.offsets[i - 1],
        };
        let stop = self.offsets[i];
        &mut self.data[start..stop]
    }
    fn iter(&self) -> IndirectIndexIter<'_, T> {
        IndirectIndexIter {
            data: self.data.as_slice(),
            offsets: self.offsets.as_slice(),
            last_offset: 0,
        }
    }
    fn iter_mut(&mut self) -> IndirectIndexIterMut<'_, T> {
        IndirectIndexIterMut {
            data: self.data.as_mut_slice(),
            offsets: self.offsets.as_mut_slice(),
            last_offset: 0,
        }
    }
    fn push(&mut self, elem: &[T])
    where
        T: Clone,
    {
        self.data.extend(elem.iter().cloned());
        self.offsets.push(self.data.len());
    }
    fn extend_from_slice(&mut self, data: &[T], offsets: &[usize])
    where
        T: Clone,
    {
        let num_elems = self.data.len();
        self.data.extend_from_slice(data);
        self.offsets.extend(offsets.iter().map(|o| o + num_elems));
    }
}

impl<'a, T> Extend<&'a [T]> for IndirectIndex<T>
where
    T: Clone,
{
    fn extend<I>(&mut self, iter: I)
    where
        I: IntoIterator<Item = &'a [T]>,
    {
        let num_elems = self.data.len();
        let (data, offsets): (Vec<&[T]>, Vec<usize>) =
            iter.into_iter().map(|sl| (sl, sl.len())).collect();
        self.data.extend(data.into_iter().flatten().cloned());
        self.offsets.extend(offsets.iter().map(|o| o + num_elems));
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

impl<'a, T> IntoIterator for &'a IndirectIndex<T> {
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

impl<'a, T> IntoIterator for &'a mut IndirectIndex<T> {
    type Item = &'a [T];
    type IntoIter = IndirectIndexIter<'a, T>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
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

impl<T> IntoIterator for IndirectIndex<T> {
    type Item = Vec<T>;
    type IntoIter = IndirectIndexIntoIter<T>;
    fn into_iter(self) -> Self::IntoIter {
        IndirectIndexIntoIter {
            data: self.data.into(),
            offsets: self.offsets.into(),
            last_offset: 0,
        }
    }
}

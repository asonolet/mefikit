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
        }
    }
}

pub struct IndirectIndexIter<'a, T>
where
    T: 'a,
{
    data: &'a [T],
    offsets: &'a [usize],
}

impl<'a, T> Iterator for IndirectIndexIter<'a, T> {
    type Item = &'a [T];

    fn next(&mut self) -> Option<Self::Item> {
        if self.offsets.is_empty() {
            return None;
        }
        let stop = self.offsets.split_off_first().unwrap();
        let data = std::mem::take(&mut self.data);
        let (chunk, rest) = data.split_at(*stop);
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
}

impl<'a, T> Iterator for IndirectIndexIterMut<'a, T> {
    type Item = &'a mut [T];

    fn next(&mut self) -> Option<Self::Item> {
        if self.offsets.is_empty() {
            return None;
        }
        let stop = self.offsets.split_off_first_mut().unwrap();
        let data = std::mem::take(&mut self.data);
        let (chunk, rest) = data.split_at_mut(*stop);
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

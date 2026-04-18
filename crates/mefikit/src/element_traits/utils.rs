use smallvec::SmallVec;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SortedVecKey(SmallVec<[usize; 4]>);

impl SortedVecKey {
    pub fn new(mut vec: SmallVec<[usize; 4]>) -> Self {
        vec.sort_unstable();
        SortedVecKey(vec)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use smallvec::smallvec;

    #[test]
    fn test_sorted_vec_key() {
        let vec1 = smallvec![3, 1, 2];
        let vec2 = smallvec![1, 2, 3];
        let key1 = SortedVecKey::new(vec1);
        let key2 = SortedVecKey::new(vec2);
        assert_eq!(key1, key2);
    }

    #[test]
    fn test_sorted_vec_key_different_order() {
        let vec1 = smallvec![5, 4, 6];
        let vec2 = smallvec![6, 5, 4];
        let key1 = SortedVecKey::new(vec1);
        let key2 = SortedVecKey::new(vec2);
        assert_eq!(key1, key2);
    }
}

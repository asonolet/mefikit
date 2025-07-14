#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SortedVecKey(Vec<usize>);

impl SortedVecKey {
    pub fn new(mut vec: Vec<usize>) -> Self {
        vec.sort_unstable();
        SortedVecKey(vec)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sorted_vec_key() {
        let vec1 = vec![3, 1, 2];
        let vec2 = vec![1, 2, 3];
        let key1 = SortedVecKey::new(vec1);
        let key2 = SortedVecKey::new(vec2);
        assert_eq!(key1, key2);
    }

    #[test]
    fn test_sorted_vec_key_different_order() {
        let vec1 = vec![5, 4, 6];
        let vec2 = vec![6, 5, 4];
        let key1 = SortedVecKey::new(vec1);
        let key2 = SortedVecKey::new(vec2);
        assert_eq!(key1, key2);
    }
}

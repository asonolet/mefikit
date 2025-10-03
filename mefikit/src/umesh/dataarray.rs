use derive_where::derive_where;
use ndarray as nd;
use serde::Serialize;

#[derive(Clone, Debug, Serialize, PartialEq)]
#[derive_where(Eq, Hash; T)]
pub enum DataArray<'a, T, D>
where
    T: 'static + Clone + std::fmt::Debug + serde::Serialize + std::cmp::PartialEq,
    D: nd::Dimension,
{
    View(nd::ArrayView<'a, T, D>),
    Shared(nd::ArcArray<T, D>),
}

impl<'a, T, D> DataArray<'a, T, D>
where
    T: 'static + Clone + std::fmt::Debug + serde::Serialize + std::cmp::PartialEq,
    D: nd::Dimension,
{
    pub fn view<'b>(&'b self) -> nd::ArrayView<'b, T, D> {
        match self {
            DataArray::View(d) => d.view(),
            DataArray::Shared(d) => d.view(),
        }
    }
    pub fn to_view(&'a self) -> nd::ArrayView<'a, T, D> {
        match self {
            DataArray::View(d) => d.clone(),
            DataArray::Shared(d) => d.view(),
        }
    }
    pub fn to_shared(&self) -> nd::ArcArray<T, D> {
        match self {
            DataArray::View(d) => d.to_shared(),
            DataArray::Shared(d) => d.to_shared(),
        }
    }
    pub fn into_shared(self) -> nd::ArcArray<T, D> {
        match self {
            DataArray::View(d) => d.to_shared(),
            DataArray::Shared(d) => d,
        }
    }
    pub fn as_slice(&self) -> Option<&[T]> {
        match self {
            DataArray::View(d) => d.as_slice(),
            DataArray::Shared(d) => d.as_slice(),
        }
    }
}
impl<'a, T> DataArray<'a, T, nd::Ix2>
where
    T: 'static + Clone + std::fmt::Debug + serde::Serialize + std::cmp::PartialEq,
{
    pub fn row(&self, index: usize) -> nd::ArrayView1<'_, T> {
        match self {
            DataArray::View(d) => d.row(index),
            DataArray::Shared(d) => d.row(index),
        }
    }
    pub fn row_slice(&'a self, index: usize) -> &'a [T] {
        match self {
            DataArray::View(d) => d.row(index).to_slice().unwrap(),
            DataArray::Shared(d) => d.row(index).to_slice().unwrap(),
        }
    }
}

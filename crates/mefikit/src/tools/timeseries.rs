//! Time-series fields.
//!
//! A time series is a set of fields sharing a common `<base>` name, each step
//! stored as an ordinary field named `<base>_step_<n>_time_<t>`. [`FieldKey`]
//! turns those names into `(base, step, time)` triples and back. Because the
//! step is part of the name, timestamps can be changed freely and two steps may
//! share a time without ever colliding.

use std::collections::BTreeMap;
use std::fmt::{Display, Formatter};

use ndarray as nd;

use crate::mesh::{Dimension, FieldBase, UMeshBase};

/// The encoded name of one step of a time series field.
///
/// `parse` accepts both the `_step_` convention (used when emitting names) and
/// the historical `_iter_` convention, so existing data keeps working.
#[derive(Debug, Clone, PartialEq)]
pub struct FieldKey {
    /// Base name shared by all the steps of the series.
    pub base: String,
    /// Step number; strictly increasing within a series.
    pub step: u64,
    /// Time stamp of the step, stored with Rust's shortest round-trip form.
    pub time: f64,
}

impl FieldKey {
    /// Creates a key for the given base, step and time.
    pub fn new(base: impl Into<String>, step: u64, time: f64) -> Self {
        Self {
            base: base.into(),
            step,
            time,
        }
    }

    /// Parses a field name into a key, or `None` when it is not a time-series
    /// step name.
    pub fn parse(name: &str) -> Option<Self> {
        let (base, rest) = LAST_MARKERS
            .iter()
            .filter_map(|&marker| name.rfind(marker).map(|idx| (idx, marker)))
            .max_by_key(|(idx, _)| *idx)
            .map(|(idx, marker)| (&name[..idx], &name[idx + marker.len()..]))?;
        if base.is_empty() {
            return None;
        }
        let (step, time) = split_step_time(rest)?;
        Some(Self {
            base: base.to_owned(),
            step,
            time,
        })
    }

    /// Encodes this key as a field name.
    pub fn name(&self) -> String {
        format!("{}_step_{}_time_{}", self.base, self.step, self.time)
    }
}

/// Markers accepted when parsing step names; the emitted form is `_step_`.
const LAST_MARKERS: [&str; 2] = ["_step_", "_iter_"];

fn split_step_time(rest: &str) -> Option<(u64, f64)> {
    let time_idx = rest.rfind("_time_")?;
    let step = rest[..time_idx].parse::<u64>().ok()?;
    let time = &rest[time_idx + "_time_".len()..];
    let time = time.parse::<f64>().ok()?;
    if !time.is_finite() {
        return None;
    }
    Some((step, time))
}

impl Display for FieldKey {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// One step of a resolved time series.
#[derive(Debug, Clone, PartialEq)]
pub struct TimeStep {
    /// Step number of this step.
    pub step: u64,
    /// Time stamp of this step.
    pub time: f64,
    /// Encoded field name holding the step's data.
    pub name: String,
}

/// Returns the step whose time is closest to `time`, prefering the lower step
/// number when two steps are equidistant.
pub fn nearest_step(steps: &[TimeStep], time: f64) -> Option<&TimeStep> {
    steps.iter().min_by(|a, b| {
        (a.time - time)
            .abs()
            .total_cmp(&(b.time - time).abs())
            .then_with(|| a.step.cmp(&b.step))
    })
}

impl<N, C, F, G> UMeshBase<N, C, F, G>
where
    N: nd::Data<Elem = f64>,
    C: nd::Data<Elem = usize>,
    F: nd::Data<Elem = f64>,
    G: nd::Data<Elem = usize>,
{
    /// Returns the time-series bases stored on this mesh, with the dimension each
    /// one resolves to.
    pub fn time_series_bases(&self) -> Vec<(String, Dimension)> {
        let mut by_base: BTreeMap<String, u8> = BTreeMap::new();
        for (et, block) in self.blocks() {
            for name in block.fields.keys() {
                if let Some(key) = FieldKey::parse(name) {
                    let dim = u8::from(et.dimension());
                    by_base
                        .entry(key.base)
                        .and_modify(|d| *d = (*d).max(dim))
                        .or_insert(dim);
                }
            }
        }
        by_base
            .into_iter()
            .filter_map(|(base, dim)| {
                let dim = Dimension::try_from(dim).ok()?;
                if self.resolve_steps_at_dim(&base, dim).is_empty() {
                    return None;
                }
                Some((base, dim))
            })
            .collect()
    }

    /// Resolves the steps of a time series, or `None` when the base has no usable
    /// steps on the requested (or highest) dimension.
    pub fn resolve_time_series(
        &self,
        base: &str,
        dim: Option<Dimension>,
    ) -> Option<(Dimension, Vec<TimeStep>)> {
        let dim = match dim {
            Some(d) => d,
            None => self
                .time_series_bases()
                .into_iter()
                .find(|(b, _)| b == base)
                .map(|(_, d)| d)?,
        };
        let steps = self.resolve_steps_at_dim(base, dim);
        if steps.is_empty() {
            return None;
        }
        Some((dim, steps))
    }

    /// Returns the step number to use for the next step of a series (`max + 1`,
    /// or `0` when the series does not exist yet).
    pub fn next_step(&self, base: &str, dim: Option<Dimension>) -> u64 {
        self.resolve_time_series(base, dim)
            .map(|(_, steps)| steps.iter().map(|s| s.step).max().unwrap_or(0) + 1)
            .unwrap_or(0)
    }

    /// Renames a field at the given dimension, moving its data.
    ///
    /// Returns `None` when the old field does not exist.
    pub fn rename_field(
        &mut self,
        old_name: &str,
        new_name: &str,
        dim: Option<Dimension>,
    ) -> Option<FieldBase<F, nd::IxDyn>> {
        let field = self.remove_field(old_name, dim)?;
        self.update_field(new_name, field)
    }

    fn resolve_steps_at_dim(&self, base: &str, dim: Dimension) -> Vec<TimeStep> {
        let mut found: BTreeMap<String, (u64, f64)> = BTreeMap::new();
        for (et, block) in self.blocks() {
            if et.dimension() != dim {
                continue;
            }
            for name in block.fields.keys() {
                if let Some(key) = FieldKey::parse(name)
                    && key.base == base
                {
                    found.insert(name.clone(), (key.step, key.time));
                }
            }
        }
        let mut steps = Vec::new();
        for (name, (step, time)) in found {
            if self.field(&name, Some(dim)).is_none() {
                continue;
            }
            steps.push(TimeStep { step, time, name });
        }
        steps.sort_by_key(|s| s.step);
        steps
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::{ElementType, FieldArcD};
    use crate::mesh_examples as me;
    use std::collections::BTreeMap;

    #[test]
    fn parse_roundtrip() {
        let key = FieldKey::new("T", 2, 0.25);
        assert_eq!(key.name(), "T_step_2_time_0.25");
        let parsed = FieldKey::parse(&key.name()).unwrap();
        assert_eq!(parsed, key);
    }

    #[test]
    fn parse_legacy_iter_convention() {
        let key = FieldKey::parse("temperature_iter_3_time_0.01").unwrap();
        assert_eq!(key.base, "temperature");
        assert_eq!(key.step, 3);
        assert_eq!(key.time, 0.01);
    }

    #[test]
    fn parse_rejects_plain_fields() {
        assert!(FieldKey::parse("T").is_none());
        assert!(FieldKey::parse("T_step_2").is_none());
        assert!(FieldKey::parse("T_iter_x_time_y").is_none());
        assert!(FieldKey::parse("T_step_x_time_y").is_none());
    }

    #[test]
    fn parse_base_with_underscores() {
        let name = "heat_flux_step_0_time_0.1";
        assert_eq!(FieldKey::parse(name).unwrap().base, "heat_flux");
        assert_eq!(FieldKey::new("heat_flux", 0, 0.1).name(), name);
    }

    #[test]
    fn nearest_prefers_the_lower_step_on_ties() {
        let steps = vec![
            TimeStep {
                step: 0,
                time: 0.0,
                name: "a".into(),
            },
            TimeStep {
                step: 1,
                time: 0.5,
                name: "b".into(),
            },
            TimeStep {
                step: 2,
                time: 0.25,
                name: "c".into(),
            },
        ];
        assert_eq!(nearest_step(&steps, 0.26).unwrap().step, 2);
        let steps = vec![
            TimeStep {
                step: 1,
                time: 0.5,
                name: "b".into(),
            },
            TimeStep {
                step: 3,
                time: 0.4,
                name: "d".into(),
            },
        ];
        assert_eq!(nearest_step(&steps, 0.45).unwrap().step, 1);
    }

    #[test]
    fn mesh_series_resolution() {
        let mut mesh = me::make_imesh_2d(2);
        let zeros = nd::ArrayD::<f64>::zeros(nd::IxDyn(&[4, 1]));
        let field = FieldArcD::new(BTreeMap::from([(ElementType::QUAD4, zeros.to_shared())]));
        mesh.update_field("T_step_0_time_0", field.clone());
        mesh.update_field("T_step_1_time_0.5", field);

        assert_eq!(
            mesh.time_series_bases(),
            vec![("T".to_owned(), Dimension::D2)]
        );
        let (dim, steps) = mesh.resolve_time_series("T", None).unwrap();
        assert_eq!(dim, Dimension::D2);
        assert_eq!(steps.len(), 2);
        assert_eq!(steps[0].step, 0);
        assert_eq!(steps[0].time, 0.0);
        assert_eq!(steps[1].step, 1);
        assert_eq!(mesh.next_step("T", None), 2);
    }
}

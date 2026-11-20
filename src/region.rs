use std::ops::Range;

use crate::Error;

/// Row-major byte strides, using a virtual leading axis of length 1 for 2D.
#[derive(Clone, Copy)]
pub(crate) struct Layout {
    row: usize,
    plane: usize,
    pub len: usize,
}

impl Layout {
    pub(crate) fn new(shape: &[u64], item_size: usize, kind: &'static str) -> Result<Self, Error> {
        if shape.contains(&0) {
            return Ok(Self {
                row: 0,
                plane: 0,
                len: 0,
            });
        }
        let [z, y, x] = padded(shape, 1);
        let multiply = |size: usize, count| {
            usize::try_from(count)
                .ok()
                .and_then(|count| size.checked_mul(count))
                .ok_or(Error::Overflow(kind))
        };
        let row = multiply(item_size, x)?;
        let plane = multiply(row, y)?;
        let len = multiply(plane, z)?;
        Ok(Self { row, plane, len })
    }

    fn offset(self, depth: usize, row: usize, column_bytes: usize) -> usize {
        depth * self.plane + row * self.row + column_bytes
    }
}

pub(crate) struct Selection {
    pub start: [u64; 3],
    pub end: [u64; 3],
    pub shape: Vec<u64>,
    pub output: Layout,
}

impl Selection {
    pub(crate) fn chunk_origins(&self, chunks: [u64; 3]) -> impl Iterator<Item = [u64; 3]> {
        let first = std::array::from_fn::<_, 3, _>(|axis| self.start[axis] / chunks[axis]);
        let last = std::array::from_fn::<_, 3, _>(|axis| (self.end[axis] - 1) / chunks[axis]);
        (first[0]..=last[0]).flat_map(move |z| {
            (first[1]..=last[1]).flat_map(move |y| {
                (first[2]..=last[2]).map(move |x| [z * chunks[0], y * chunks[1], x * chunks[2]])
            })
        })
    }

    pub(crate) fn new(
        ranges: &[Range<u64>],
        extents: &[u64],
        item_size: usize,
    ) -> Result<Self, Error> {
        if ranges.len() != extents.len() {
            return Err(Error::SelectionRank {
                expected: extents.len(),
                actual: ranges.len(),
            });
        }
        let mut start = [0; 3];
        let mut end = [1; 3];
        let mut shape = Vec::with_capacity(extents.len());
        for (axis, (range, &extent)) in ranges.iter().zip(extents).enumerate() {
            if range.start > range.end || range.end > extent {
                return Err(Error::SelectionBounds {
                    axis,
                    start: range.start,
                    end: range.end,
                    extent,
                });
            }
            let normalized = 3 - extents.len() + axis;
            start[normalized] = range.start;
            end[normalized] = range.end;
            shape.push(range.end - range.start);
        }
        let output = Layout::new(&shape, item_size, "output")?;
        Ok(Self {
            start,
            end,
            shape,
            output,
        })
    }

    pub(crate) fn copy_chunk(
        &self,
        origin: [u64; 3],
        chunk_shape: [u64; 3],
        chunk_layout: Layout,
        data: Option<&[u8]>,
        fill: &[u8],
        output: &mut [u8],
    ) {
        let begin = std::array::from_fn::<_, 3, _>(|axis| origin[axis].max(self.start[axis]));
        let end = std::array::from_fn::<_, 3, _>(|axis| {
            origin[axis]
                .saturating_add(chunk_shape[axis])
                .min(self.end[axis])
        });
        // Both layouts have been checked for overflow. Differences within an
        // intersection fit their corresponding layout strides.
        let width = (end[2] - begin[2]) as usize * fill.len();
        let dst_column = (begin[2] - self.start[2]) as usize * fill.len();
        let src_column = (begin[2] - origin[2]) as usize * fill.len();
        for z in begin[0]..end[0] {
            for y in begin[1]..end[1] {
                let dst = self.output.offset(
                    (z - self.start[0]) as usize,
                    (y - self.start[1]) as usize,
                    dst_column,
                );
                let row = &mut output[dst..dst + width];
                if let Some(data) = data {
                    let src = chunk_layout.offset(
                        (z - origin[0]) as usize,
                        (y - origin[1]) as usize,
                        src_column,
                    );
                    row.copy_from_slice(&data[src..src + width]);
                } else {
                    for element in row.chunks_exact_mut(fill.len()) {
                        element.copy_from_slice(fill);
                    }
                }
            }
        }
    }
}

pub(crate) fn padded(shape: &[u64], leading: u64) -> [u64; 3] {
    let mut normalized = [leading; 3];
    normalized[3 - shape.len()..].copy_from_slice(shape);
    normalized
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checks_size_overflow_and_preserves_empty_shape() {
        let ranges = [0..u64::MAX, 0..2];
        let extents = [u64::MAX, 2];
        assert!(matches!(
            Selection::new(&ranges, &extents, 8),
            Err(Error::Overflow("output"))
        ));
        let selection = Selection::new(&[0..u64::MAX, 2..2], &extents, 8).unwrap();
        assert_eq!(selection.shape, [u64::MAX, 0]);
        assert_eq!(selection.output.len, 0);
    }
}

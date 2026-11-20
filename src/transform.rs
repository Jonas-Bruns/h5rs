use rayon::prelude::*;

use crate::{error::zeroed, region::Layout, ByteOrder, DType, Error, NumericType, Region};

pub(crate) fn stride(input: Region, steps: &[isize]) -> Result<Region, Error> {
    if steps.iter().all(|&step| step == 1) {
        return Ok(input);
    }
    let mut source_shape = [1_u64; 3];
    let mut stride = [1; 3];
    let rank = input.shape.len();
    for (axis, (&length, &step)) in input.shape.iter().zip(steps).enumerate() {
        source_shape[3 - rank + axis] = length;
        stride[3 - rank + axis] = step;
    }
    let shape = std::array::from_fn::<_, 3, _>(|axis| {
        source_shape[axis].div_ceil(stride[axis].unsigned_abs() as u64)
    });
    let width = input.dtype.size();
    let mut bytes = zeroed(Layout::new(&shape, width, "strided output")?.len)?;
    if bytes.is_empty() {
        return Ok(Region {
            shape: shape[3 - rank..].to_vec(),
            dtype: input.dtype,
            bytes,
        });
    }
    let source_shape = source_shape.map(|size| size as usize);
    let index = |axis: usize, position: usize| {
        let distance = position * stride[axis].unsigned_abs();
        if stride[axis] < 0 {
            source_shape[axis] - 1 - distance
        } else {
            distance
        }
    };
    let row_bytes = shape[2] as usize * width;
    let rows = shape[1] as usize;
    bytes
        .par_chunks_mut(row_bytes)
        .enumerate()
        .for_each(|(row, output)| {
            let z = index(0, row / rows);
            let y = index(1, row % rows);
            let start = (z * source_shape[1] + y) * source_shape[2] * width;
            if stride[2] == 1 {
                output.copy_from_slice(&input.bytes[start..start + row_bytes]);
            } else {
                for (column, element) in output.chunks_exact_mut(width).enumerate() {
                    let offset = start + index(2, column) * width;
                    element.copy_from_slice(&input.bytes[offset..offset + width]);
                }
            }
        });
    Ok(Region {
        shape: shape[3 - rank..].to_vec(),
        dtype: input.dtype,
        bytes,
    })
}

pub(crate) fn min_max(input: Region, axis: usize, block: usize, mut output: Region) -> Region {
    if output.bytes.is_empty() {
        return output;
    }
    let inner = input.shape[axis + 1..].iter().product::<u64>() as usize;
    let axis_len = input.shape[axis] as usize;
    let bins = axis_len.div_ceil(block);
    let width = input.dtype.size();
    macro_rules! reduce {
        ($type:ty) => {{
            let little = input.dtype.byte_order != ByteOrder::Big;
            let read = |index: usize| {
                let offset = index * width;
                let bytes = input.bytes[offset..offset + width]
                    .try_into()
                    .expect("numeric width");
                if little {
                    <$type>::from_le_bytes(bytes)
                } else {
                    <$type>::from_be_bytes(bytes)
                }
            };
            output
                .bytes
                .par_chunks_exact_mut(2 * width)
                .enumerate()
                .for_each(|(index, pair)| {
                    let outer = index / (inner * bins);
                    let bin = index / inner % bins;
                    let start = bin * block;
                    let end = start.saturating_add(block).min(axis_len);
                    let source = (outer * axis_len + start) * inner + index % inner;
                    let mut low = read(source);
                    let mut high = low;
                    for offset in 1..end - start {
                        let value = read(source + offset * inner);
                        low = low.min(value);
                        high = high.max(value);
                    }
                    let (left, right) = pair.split_at_mut(width);
                    if little {
                        left.copy_from_slice(&low.to_le_bytes());
                        right.copy_from_slice(&high.to_le_bytes());
                    } else {
                        left.copy_from_slice(&low.to_be_bytes());
                        right.copy_from_slice(&high.to_be_bytes());
                    }
                });
        }};
    }
    use NumericType::*;
    match input.dtype.kind {
        I8 => reduce!(i8),
        I16 => reduce!(i16),
        I32 => reduce!(i32),
        I64 => reduce!(i64),
        U8 => reduce!(u8),
        U16 => reduce!(u16),
        U32 => reduce!(u32),
        U64 => reduce!(u64),
        F32 => reduce!(f32),
        F64 => reduce!(f64),
    }
    output
}

pub(crate) fn f32(input: Region, mut bytes: Vec<u8>) -> Region {
    let width = input.dtype.size();
    let little = input.dtype.byte_order != ByteOrder::Big;
    macro_rules! cast {
        ($type:ty) => {
            bytes
                .par_chunks_exact_mut(4)
                .zip(input.bytes.par_chunks_exact(width))
                .for_each(|(target, source)| {
                    let source = source.try_into().expect("numeric width");
                    let value = if little {
                        <$type>::from_le_bytes(source)
                    } else {
                        <$type>::from_be_bytes(source)
                    };
                    target.copy_from_slice(&(value as f32).to_le_bytes());
                })
        };
    }
    use NumericType::*;
    match input.dtype.kind {
        I8 => cast!(i8),
        I16 => cast!(i16),
        I32 => cast!(i32),
        I64 => cast!(i64),
        U8 => cast!(u8),
        U16 => cast!(u16),
        U32 => cast!(u32),
        U64 => cast!(u64),
        F32 => cast!(f32),
        F64 => cast!(f64),
    }
    Region {
        shape: input.shape,
        dtype: DType {
            kind: F32,
            byte_order: ByteOrder::Little,
        },
        bytes,
    }
}

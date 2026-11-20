use std::borrow::Cow;

use flate2::{Decompress, FlushDecompress, Status};

use crate::{error::zeroed, Error, Filter};

pub(crate) fn decode<'a>(
    mut data: Cow<'a, [u8]>,
    mask: u32,
    filters: &[Filter],
    decoded_size: usize,
    item_size: usize,
) -> Result<Cow<'a, [u8]>, Error> {
    for (index, filter) in filters.iter().enumerate().rev() {
        if mask & (1 << index) != 0 {
            continue;
        }
        data = match filter {
            Filter::Deflate { .. } => Cow::Owned(inflate(&data, decoded_size)?),
            Filter::Shuffle if item_size > 1 => {
                check_size(&data, decoded_size)?;
                let elements = decoded_size / item_size;
                let mut output = zeroed(decoded_size)?;
                for (byte, plane) in data.chunks_exact(elements).enumerate() {
                    for (element, value) in plane.iter().enumerate() {
                        output[element * item_size + byte] = *value;
                    }
                }
                Cow::Owned(output)
            }
            Filter::Shuffle => data,
        };
    }
    check_size(&data, decoded_size)?;
    Ok(data)
}

fn check_size(data: &[u8], decoded_size: usize) -> Result<(), Error> {
    if data.len() != decoded_size {
        return Err(Error::InvalidChunk("unexpected decoded size"));
    }
    Ok(())
}

fn inflate(data: &[u8], decoded_size: usize) -> Result<Vec<u8>, Error> {
    // One extra byte lets the inflater reach the trailer and detects excess output.
    let capacity = decoded_size
        .checked_add(1)
        .ok_or(Error::Overflow("inflate buffer"))?;
    let mut output = zeroed(capacity)?;
    let mut decoder = Decompress::new(true);
    let status = decoder.decompress(data, &mut output, FlushDecompress::Finish)?;
    if status != Status::StreamEnd
        || decoder.total_out() != decoded_size as u64
        || decoder.total_in() != data.len() as u64
    {
        return Err(Error::InvalidChunk("unexpected deflate stream length"));
    }
    output.truncate(decoded_size);
    Ok(output)
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use flate2::{write::ZlibEncoder, Compression};

    use super::*;

    #[test]
    fn validates_zlib_length_and_trailer() {
        let source = [7; 64];
        let mut encoder = ZlibEncoder::new(Vec::new(), Compression::fast());
        encoder.write_all(&source).unwrap();
        let mut data = encoder.finish().unwrap();
        assert_eq!(inflate(&data, source.len()).unwrap(), source);
        assert!(inflate(&data, source.len() + 1).is_err());
        assert!(inflate(&data, source.len() - 1).is_err());
        assert!(inflate(&data[..data.len() - 1], source.len()).is_err());
        data.push(0);
        assert!(inflate(&data, source.len()).is_err());
    }
}

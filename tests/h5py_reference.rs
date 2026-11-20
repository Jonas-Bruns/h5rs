use std::fs;
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use h5rs::{ByteOrder, DType, Dataset, Error, Filter, H5File, Limits, NumericType, ReadPool};

#[test]
#[ignore = "requires Python with h5py >= 3.16 and NumPy"]
fn selections_match_h5py() {
    let fixtures = generate_fixtures();
    let pool = ReadPool::new(4, 3).unwrap();
    let file = H5File::open_with_pool(
        fixtures.0.join("reference.h5"),
        Limits::default(),
        pool.clone(),
    )
    .unwrap();
    let latest =
        H5File::open_with_pool(fixtures.0.join("latest.h5"), Limits::default(), pool).unwrap();
    let cases = fs::read_to_string(fixtures.0.join("cases.tsv")).unwrap();

    for line in cases.lines() {
        let fields: Vec<_> = line.split('\t').collect();
        let [source, name, dtype, shape, chunks, region_shape, starts, ends, oracle] =
            fields.as_slice()
        else {
            panic!("unexpected oracle case {line}");
        };
        let dataset = if *source == "reference.h5" {
            &file
        } else {
            &latest
        }
        .dataset(name)
        .unwrap_or_else(|err| panic!("{source} {name}: {err}"));

        let ranges: Vec<_> = dimensions(starts)
            .into_iter()
            .zip(dimensions(ends))
            .map(|(start, end)| start..end)
            .collect();
        assert_eq!(dataset.metadata().shape, dimensions(shape), "{line}");
        assert_eq!(dataset.metadata().chunks, dimensions(chunks), "{line}");
        let region = dataset
            .read_region(&ranges)
            .unwrap_or_else(|err| panic!("{line}: {err}"));
        assert_eq!(region.shape, dimensions(region_shape), "{line}");
        assert_eq!(region.dtype, numpy_dtype(dtype), "{line}");
        let expected = fs::read(fixtures.0.join(format!("{oracle}.bin"))).unwrap();
        assert_eq!(region.bytes, expected, "{line}");

        let mut supplied = vec![0xff; expected.len()];
        dataset.read_region_into(&ranges, &mut supplied).unwrap();
        assert_eq!(supplied, expected, "{line}");
    }

    assert_errors(&fixtures.0, &file);
    assert_concurrent_reads(&file, &latest);
    let large = H5File::open(fixtures.0.join("large-index.h5")).unwrap();
    let large = large.dataset("large").unwrap();
    assert_eq!(
        large.read_region(&[256..257, 254..256]).unwrap().bytes,
        [65_790_u32.to_le_bytes(), 65_791_u32.to_le_bytes()].concat()
    );

    let asynchronous = pollster::block_on(async {
        let file = H5File::open_async(fixtures.0.join("reference.h5"))
            .await
            .unwrap();
        let dataset = file.dataset_async("mixed").await.unwrap();
        dataset.read_region_async(&[0..11, 0..17]).await.unwrap()
    });
    let expected = file
        .dataset("mixed")
        .unwrap()
        .read_region(&[0..11, 0..17])
        .unwrap();
    assert_eq!(asynchronous.bytes, expected.bytes);
}

#[cfg(feature = "s3")]
#[test]
#[ignore = "requires Python with h5py and a loopback HTTP listener"]
fn s3_ranges_read_hdf5_and_overlap() {
    use std::thread;
    use std::time::{Duration, Instant};

    let fixtures = generate_fixtures();
    let server_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/range_server.py");
    let mut server = RangeServer(
        Command::new(python())
            .arg(server_path)
            .arg(&fixtures.0)
            .spawn()
            .unwrap(),
    );
    let start = Instant::now();
    let port = loop {
        if let Ok(port) = fs::read_to_string(fixtures.0.join("port")) {
            if let Ok(port) = port.parse::<u16>() {
                break port;
            }
        }
        if let Some(status) = server.0.try_wait().expect("check range server process") {
            panic!("range server exited before reporting its port: {status}");
        }
        assert!(
            start.elapsed() < Duration::from_secs(5),
            "range server did not start"
        );
        thread::sleep(Duration::from_millis(20));
    };
    let endpoint = format!("http://127.0.0.1:{port}");
    let pool = ReadPool::new(4, 2).unwrap();
    let first = H5File::open_s3_with_pool(
        &format!("{endpoint}/reference"),
        Limits::default(),
        pool.clone(),
    )
    .unwrap();
    let second =
        H5File::open_s3_with_pool(&format!("{endpoint}/latest"), Limits::default(), pool).unwrap();
    let latest = H5File::open(fixtures.0.join("latest.h5")).unwrap();

    let raw = first.dataset("raw").unwrap();
    let mixed = first.dataset("mixed").unwrap();
    let extendible = second.dataset("extendible").unwrap();
    assert_eq!(
        raw.read_region(&[0..11, 0..17]).unwrap().bytes,
        fs::read(fixtures.0.join("0.bin")).unwrap()
    );
    let counters = fs::read_to_string(fixtures.0.join("chunk-reads"))
        .unwrap()
        .split_whitespace()
        .map(|value| value.parse::<usize>().unwrap())
        .collect::<Vec<_>>();
    let [peak, reads, read_bytes] = counters.as_slice() else {
        panic!("unexpected range counters {counters:?}");
    };
    assert!(
        *peak <= 4,
        "range concurrency exceeds the worker count ({peak})"
    );
    assert!(
        *reads < 9,
        "adjacent chunks should share requests ({reads})"
    );
    assert_eq!(
        *read_bytes,
        9 * 4 * 6 * 4,
        "read only the stored chunk bytes"
    );
    // A single-column chunk selection leaves gaps on disk. Those reads
    // should still overlap even when the full selection can use one GET.
    let expected_column = H5File::open(fixtures.0.join("reference.h5"))
        .unwrap()
        .dataset("raw")
        .unwrap()
        .read_region(&[0..11, 0..6])
        .unwrap();
    assert_eq!(
        raw.read_region(&[0..11, 0..6]).unwrap().bytes,
        expected_column.bytes
    );
    let peak: usize = fs::read_to_string(fixtures.0.join("chunk-reads"))
        .unwrap()
        .split_whitespace()
        .next()
        .unwrap()
        .parse()
        .unwrap();
    assert!(
        (2..=4).contains(&peak),
        "expected overlapping reads within the worker limit ({peak})"
    );
    assert_eq!(
        first
            .dataset("gzip")
            .unwrap()
            .read_region(&[0..7, 0..11, 0..9])
            .unwrap()
            .bytes,
        fs::read(fixtures.0.join("18.bin")).unwrap(),
    );
    let expected = latest
        .dataset("extendible")
        .unwrap()
        .read_region(&[1..13, 2..7])
        .unwrap();
    assert_eq!(
        extendible.read_region(&[1..13, 2..7]).unwrap().bytes,
        expected.bytes
    );
    let async_data = pollster::block_on(async {
        let file = H5File::open_s3_async(&format!("{endpoint}/reference?signature=mock"))
            .await
            .unwrap();
        file.dataset_async("mixed")
            .await
            .unwrap()
            .read_region_async(&[1..10, 2..16])
            .await
            .unwrap()
    });
    assert_eq!(
        async_data.bytes,
        mixed.read_region(&[1..10, 2..16]).unwrap().bytes
    );

    let ranges = [1..10, 2..16];
    let expected = raw.read_region(&ranges).unwrap().bytes;
    let indexed = first.dataset_with_s3_index("raw").unwrap();
    for _ in 0..2 {
        assert_eq!(indexed.read_region(&ranges).unwrap().bytes, expected);
    }
    assert!(matches!(
        latest.dataset_with_s3_index("extendible"),
        Err(Error::Unsupported("S3 index requires an S3 object"))
    ));
    let indexed = first.dataset_with_s3_index("mixed").unwrap();
    assert_eq!(
        indexed.read_region(&ranges).unwrap().bytes,
        async_data.bytes
    );

    let large_file = H5File::open_s3(&format!("{endpoint}/large-index")).unwrap();
    let large = large_file.dataset_with_s3_index("large").unwrap();
    assert_eq!(
        large.read_region(&[256..257, 254..256]).unwrap().bytes,
        [65_790_u32.to_le_bytes(), 65_791_u32.to_le_bytes()].concat()
    );
    let sparse = first.dataset_with_s3_index("high").unwrap();
    let end = 1_u64 << 40;
    assert_eq!(
        sparse.read_region(&[end - 3..end, 0..2]).unwrap().bytes,
        [0, 0, 7, 8, 9, 10],
    );

    let changed = H5File::open_s3(&format!("{endpoint}/changed")).unwrap();
    let changed = changed.dataset_with_s3_index("raw").unwrap();
    fs::write(fixtures.0.join("changed"), "").unwrap();
    assert!(matches!(
        changed.read_region(&[0..1, 0..1]),
        Err(Error::ObjectChanged)
    ));
    for mode in ["ignored", "short", "weak"] {
        assert!(
            matches!(
                H5File::open_s3(&format!("{endpoint}/{mode}")),
                Err(Error::InvalidRangeResponse(_))
            ),
            "{mode}"
        );
    }
    assert!(matches!(
        H5File::open_s3(&format!("{endpoint}/bad-metadata")),
        Err(Error::RemoteStatus(503))
    ));
}

#[cfg(feature = "s3")]
struct RangeServer(std::process::Child);

#[cfg(feature = "s3")]
impl Drop for RangeServer {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn assert_errors(path: &Path, file: &H5File) {
    for name in ["text", "contiguous", "rank4", "never"] {
        assert!(
            matches!(file.dataset(name), Err(Error::Unsupported(_))),
            "{name}"
        );
    }
    assert!(matches!(
        file.dataset("fletcher"),
        Err(Error::UnsupportedFilter(3))
    ));
    assert!(matches!(
        file.dataset("external"),
        Err(Error::Hdf5("open dataset"))
    ));
    assert!(matches!(
        file.dataset("missing"),
        Err(Error::Hdf5("open dataset"))
    ));
    assert!(matches!(
        file.dataset("invalid\0name"),
        Err(Error::InvalidName)
    ));
    let dataset = file.dataset("mixed").unwrap();
    assert_eq!(
        dataset.metadata().filters,
        [Filter::Shuffle, Filter::Deflate { level: 6 }]
    );
    assert_eq!(dataset.metadata().fill_value, (-2345_i16).to_be_bytes());
    assert!(matches!(
        dataset.read_region(&[]),
        Err(Error::SelectionRank { .. })
    ));
    assert!(matches!(
        dataset.read_region(&[0..12, 0..1]),
        Err(Error::SelectionBounds { .. })
    ));
    let reversed = Range { start: 3, end: 2 };
    assert!(matches!(
        dataset.read_region(&[reversed, 0..1]),
        Err(Error::SelectionBounds { .. })
    ));
    assert!(matches!(
        dataset.read_region_into(&[0..1, 0..1], &mut [0; 3]),
        Err(Error::OutputLength {
            expected: 2,
            actual: 3
        })
    ));

    let bounded = H5File::open_with_limits(
        path.join("reference.h5"),
        Limits {
            max_output_bytes: 2,
            max_chunk_bytes: 48,
        },
    )
    .unwrap();
    assert!(matches!(
        bounded.dataset("raw"),
        Err(Error::Limit {
            kind: "decoded chunk",
            ..
        })
    ));
    let bounded = bounded.dataset("mixed").unwrap();
    assert!(matches!(
        bounded.read_region(&[0..2, 0..2]),
        Err(Error::Limit { kind: "output", .. })
    ));

    assert!(H5File::open(path.join("truncated.h5")).is_err());
    let corrupt = H5File::open(path.join("corrupt.h5")).unwrap();
    assert!(matches!(
        corrupt.dataset("data").unwrap().read_region(&[0..1, 0..1]),
        Err(Error::Deflate(_) | Error::InvalidChunk(_))
    ));
}

fn assert_concurrent_reads(file: &H5File, latest: &H5File) {
    let raw = file.dataset("raw").unwrap();
    let gzip = file.dataset("gzip").unwrap();
    let mixed = file.dataset("mixed").unwrap();
    let extendible = latest.dataset("extendible").unwrap();
    let reads: [(&Dataset, &[Range<u64>]); 4] = [
        (&raw, &[1..11, 2..16]),
        (&gzip, &[1..7, 1..10, 2..9]),
        (&mixed, &[2..10, 3..16]),
        (&extendible, &[1..13, 2..7]),
    ];
    let expected = reads.map(|(dataset, ranges)| dataset.read_region(ranges).unwrap().bytes);
    std::thread::scope(|scope| {
        let expected = &expected;
        for worker in 0..8 {
            scope.spawn(move || {
                for request in 0..8 {
                    let case = (worker + request) % reads.len();
                    let (dataset, ranges) = reads[case];
                    assert_eq!(dataset.read_region(ranges).unwrap().bytes, expected[case]);
                }
            });
        }
    });
}

fn dimensions(field: &str) -> Vec<u64> {
    field
        .split(',')
        .map(|value| value.parse().unwrap())
        .collect()
}

fn numpy_dtype(dtype: &str) -> DType {
    use NumericType::*;
    let (order, tag) = dtype.split_at(1);
    let byte_order = match order {
        "|" => ByteOrder::NotApplicable,
        "<" => ByteOrder::Little,
        ">" => ByteOrder::Big,
        _ => panic!("unexpected byte order {order}"),
    };
    let kind = match tag {
        "i1" => I8,
        "u1" => U8,
        "i2" => I16,
        "u2" => U16,
        "i4" => I32,
        "u4" => U32,
        "i8" => I64,
        "u8" => U64,
        "f4" => F32,
        "f8" => F64,
        _ => panic!("unexpected NumPy dtype {dtype}"),
    };
    DType { kind, byte_order }
}

fn generate_fixtures() -> Fixtures {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let path = std::env::temp_dir().join(format!("h5rs-{}-{nonce}", std::process::id()));
    fs::create_dir(&path).unwrap();
    let fixtures = Fixtures(path);
    let generator = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/reference.py");
    let status = Command::new(python())
        .arg(generator)
        .arg(&fixtures.0)
        .status()
        .expect("could not launch Python; set H5RS_TEST_PYTHON to an interpreter with h5py");
    assert!(status.success(), "h5py fixture generation failed");
    fixtures
}

fn python() -> std::ffi::OsString {
    std::env::var_os("H5RS_TEST_PYTHON").unwrap_or_else(|| "python".into())
}

struct Fixtures(PathBuf);

impl Drop for Fixtures {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
